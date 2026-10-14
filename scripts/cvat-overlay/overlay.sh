#!/usr/bin/env bash
# Inject/remove the 3D projection overlay into the running CVAT instance.
#
#   scripts/cvat-overlay/overlay.sh up   --calib calib.json --pcd-dir data/pointcloud
#   scripts/cvat-overlay/overlay.sh down
#   scripts/cvat-overlay/overlay.sh status
#
# When "up", the JS plugin is copied into cvat_ui and nginx is reloaded.
# Reload the CVAT tab (Ctrl+Shift+R) to see boxes projected onto camera images.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CVAT_URL="${CVAT_URL:-http://localhost:8080}"
UI="${CVAT_UI_CONTAINER:-cvat_ui}"

CONF=/etc/nginx/conf.d/default.conf
STOCK=/etc/nginx/conf.d/default.conf.d14-stock
WWW=/usr/share/nginx/d14-overlay
PRIVATE=/usr/share/nginx/d14-overlay-private

die() { echo "✗ $*" >&2; exit 1; }
ui() { docker exec "$UI" "$@"; }
ui_root() { docker exec -u 0 "$UI" "$@"; }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$CVAT_URL$1" || true; }

preflight() {
  command -v docker >/dev/null || die "docker not found"
  command -v python3 >/dev/null || die "python3 not found"
  docker info >/dev/null 2>&1 \
    || die "cannot reach Docker; start Docker first"
  [ "$(docker inspect -f '{{.State.Running}}' "$UI" 2>/dev/null)" = "true" ] \
    || die "container $UI is not running; start CVAT first"
}

stock_conf() {
  if ! ui test -f "$STOCK"; then
    ui grep -q d14-overlay "$CONF" \
      && die "$CONF already has the overlay but stock copy is gone; recreate $UI"
    ui_root cp -p "$CONF" "$STOCK"
  fi
  ui cat "$STOCK"
}

nginx_ok() {
  local out
  out="$(ui nginx "$@" 2>&1)" && return 0
  echo "$out" >&2
  return 1
}

reload_or_restore() {
  if ! nginx_ok -t; then
    ui_root cp -p "$STOCK" "$CONF"
    die "generated nginx config rejected; $UI keeps stock config"
  fi
  nginx_ok -s reload || die "nginx reload failed in $UI"
}

# parse --calib/--pcd-dir/--cameras/--task-ids into globals A_CALIB/A_PCD/A_CAMERAS/A_TASKS
parse_build_args() {
  A_CALIB=""; A_PCD=""; A_CAMERAS=""; A_TASKS=""
  shift  # skip subcommand
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --calib)    A_CALIB="$2"; shift 2 ;;
      --pcd-dir)  A_PCD="$2"; shift 2 ;;
      --cameras)  A_CAMERAS="$2"; shift 2 ;;
      --task-ids) A_TASKS="$2"; shift 2 ;;
      *) die "unknown flag: $1" ;;
    esac
  done
  [ -n "$A_CALIB" ] || die "usage: $0 <up|register> --calib <calib.json> --pcd-dir <pointcloud/> [--cameras <json>] [--task-ids <id,..>]"
  [ -n "$A_PCD" ] || die "usage: $0 <up|register> --calib <calib.json> --pcd-dir <pointcloud/> [--cameras <json>] [--task-ids <id,..>]"
}

# install plugin JS + nginx hook + the given config.json into cvat_ui (idempotent)
install_plugin() {
  local config_file="$1"
  stock_conf >"$tmp/stock.conf"
  python3 "$HERE/overlay-config.py" nginx --stock "$tmp/stock.conf" --out "$tmp/default.conf"

  ui_root mkdir -p "$WWW" "$PRIVATE"
  docker cp "$HERE/d14-projection-overlay.js" "$UI:$WWW/d14-projection-overlay.js"
  docker cp "$config_file" "$UI:$PRIVATE/config.json"
  docker cp "$tmp/default.conf" "$UI:$CONF"
  ui_root chown -R nginx:nginx "$WWW" "$PRIVATE" "$CONF"
  ui_root chmod 755 "$WWW"
  ui_root chmod 644 "$WWW/d14-projection-overlay.js" "$CONF"
  ui_root chmod 700 "$PRIVATE"
  ui_root chmod 600 "$PRIVATE/config.json"
  reload_or_restore
}

up() {
  parse_build_args "$@"
  preflight
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  python3 "$HERE/overlay-config.py" config \
    --calib "$A_CALIB" --pcd-dir "$A_PCD" \
    ${A_CAMERAS:+--cameras "$A_CAMERAS"} ${A_TASKS:+--task-ids "$A_TASKS"} \
    --out "$tmp/config.json"
  install_plugin "$tmp/config.json"
  sleep 1
  status
  echo "→ reload the CVAT tab (Ctrl+Shift+R) to load the plugin"
}

# merge one dataset into the live registry, keeping the others (used per task)
register() {
  parse_build_args "$@"
  preflight
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  # current registry from the container (empty if none / plugin not yet installed)
  ui cat "$PRIVATE/config.json" >"$tmp/existing.json" 2>/dev/null || : >"$tmp/existing.json"
  python3 "$HERE/overlay-config.py" register \
    --calib "$A_CALIB" --pcd-dir "$A_PCD" \
    ${A_CAMERAS:+--cameras "$A_CAMERAS"} ${A_TASKS:+--task-ids "$A_TASKS"} \
    --existing "$tmp/existing.json" --out "$tmp/config.json"
  install_plugin "$tmp/config.json"
  sleep 1
  status
}

down() {
  preflight
  if ui test -f "$STOCK"; then
    ui_root cp -p "$STOCK" "$CONF"
    nginx_ok -t || die "stock nginx config rejected; recreate $UI"
    nginx_ok -s reload || die "nginx reload failed"
  fi
  ui_root rm -rf "$WWW" "$PRIVATE" "$STOCK"
  sleep 1
  status
}

status() {
  local about version page
  about="$(curl -s --max-time 5 "$CVAT_URL/api/server/about" || true)"
  version="$(python3 -c 'import json, sys
try: print(json.load(sys.stdin).get("version", "?"))
except ValueError: print("unreachable")' <<<"$about")"
  echo "CVAT at $CVAT_URL: $version"
  page="$(curl -s --max-time 5 "$CVAT_URL/" || true)"
  if ! grep -q 'd14-projection-overlay.js' <<<"$page"; then
    echo "overlay: off"
    return 0
  fi
  echo "overlay: on"
  echo "  plugin: HTTP $(code /d14-overlay/d14-projection-overlay.js) (expect 200)"
  echo "  config: HTTP $(code /d14-overlay/private/config.json) (expect 401 without login)"
  ui cat "$PRIVATE/config.json" 2>/dev/null | python3 -c 'import json, sys
c = json.load(sys.stdin)
ds = c["datasets"] if isinstance(c, dict) and isinstance(c.get("datasets"), list) else [c]
print("  datasets: %d" % len(ds))
for d in ds:
    v = d.get("calibration_version", {})
    print("    - %s sha256 %s, %d frames, tasks %s, cameras %s"
          % (v.get("file", "?"), str(v.get("sha256", ""))[:12], len(d.get("frames", [])),
             d.get("task_ids", []), list((d.get("cameras") or {}).keys())))' \
    || echo "  calibration: config missing in $UI"
}

case "${1:-}" in
  up) up "$@" ;;
  register) register "$@" ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up|register --calib <calib.json> --pcd-dir <pointcloud/> [--cameras <json>] [--task-ids <id,..>]"
     echo "       $0 down|status"; exit 1 ;;
esac
