#!/bin/bash
# Run before the real test (20/11, about 11:50), from this folder:      bash preflight.sh
#  1. shows the settings and lets you set the MQTT event for the test
#  2. moves the rehearsal results into rehearsal_<time>/ (so the test zip holds only the test)
#  3. tunes the network (asks for the Pi password)
#  4. checks the server login, the remaining test attempts and the MQTT login
cd "$(dirname "$0")" || exit 1
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ~/TGR-21_day2/venv/bin/activate ]; then source ~/TGR-21_day2/venv/bin/activate; fi
PY=python; command -v python >/dev/null || PY=python3
echo "== 1. settings"
$PY - <<'PYEOF'
import json
c = json.load(open("team_config.json"))
print(f"  server : {c.get('HTTP_URL') or 'NOT SET'}")
print(f"  team   : {c.get('TEAM_ID')}   website password saved: {'yes' if c.get('TEAM_PASSWORD') else 'NO'}   MQTT key saved: {'yes' if c.get('MQTT_KEY') else 'NO'}")
print(f"  MQTT   : topic /event/<key>/{c.get('TRACK')}/{c.get('EVENT')}      (event now: {c.get('EVENT')})")
PYEOF
read -r -p "  MQTT event for the test (press Enter to keep the one shown): " ev
if [ -n "$ev" ]; then
  $PY -c "import json,sys;p='team_config.json';c=json.load(open(p));c['EVENT']=sys.argv[1];json.dump(c,open(p,'w'),indent=1);print('  event set to',sys.argv[1])" "$ev"
fi
echo "== 2. rehearsal results"
if [ -e test_data ] || [ -e logs ] || [ -n "$(ls -A outputs 2>/dev/null | grep -v '^.keep$')" ]; then
  b="rehearsal_$(date +%H%M%S)"; mkdir -p "$b"
  for d in test_data outputs logs; do [ -e "$d" ] && mv "$d" "$b/"; done
  mkdir -p outputs; touch outputs/.keep; echo "  moved to $b/"
else
  echo "  nothing to move"
fi
echo "== 3. network"
sudo bash net_tune.sh
echo "== 4. server and MQTT login"
$PY client.py --check
echo
echo "Ready if step 4 shows \"remaining_attempts\": 5 and \"MQTT login: OK\"."
echo "At 12:00:    python client.py --mode test      (it asks you to type YES; run it inside tmux)"
echo "Before 12:00 (no outputs in it, not sent over HTTP): python client.py --make-zip  -> submit TGR-21_Day3.zip"
echo "After the test: python check_outputs.py test_data outputs"
