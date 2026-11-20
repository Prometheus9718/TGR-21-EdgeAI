#!/bin/bash
# Speed settings for timed runs. Run right before a run, never while client.py is running (open connections
# would break). They reset at reboot.        sudo bash net_tune.sh
#  1. Internet path: downloads 2 MB from Cloudflare (the network the competition server sits behind) twice
#     through every interface that has internet (cable / Wi-Fi) and sends internet traffic through the fastest.
#     The local network (SSH from the Mac) is not touched.
#  2. TCP keeps its sending rate after a pause; a new connection may send ~58 KB in its first round trip.
#  3. The CPU stays at full clock while the photo downloads.
set -u
URL="https://speed.cloudflare.com/__down?bytes=2000000"
sysctl -q -w net.ipv4.tcp_slow_start_after_idle=0 && echo "TCP: keeps its rate after pauses"
# Wi-Fi power saving makes the chip doze between beacons (~100 ms): every round trip can wait for it. Turn it off.
for w in $(iw dev 2>/dev/null | awk '$1 == "Interface" {print $2}'); do
  iw dev "$w" set power_save off 2>/dev/null; echo "$w: $(iw dev "$w" get power_save 2>/dev/null)"
done
while ip route del default metric 50 2>/dev/null; do :; done        # undo this script's previous choice
best_dev=""; best_gw=""; best_t=""
while read -r gw dev; do
  [ -z "$dev" ] && continue
  t_min=""
  for n in 1 2; do
    out=$(curl -s -o /dev/null --max-time 15 --interface "$dev" -w "%{size_download} %{time_connect} %{time_total}" "$URL" 2>/dev/null)
    if [ "${out%% *}" = "2000000" ]; then
      t=${out##* }; rtt=$(echo "$out" | awk '{print $2}')
      if [ -z "$t_min" ] || awk -v a="$t" -v b="$t_min" 'BEGIN{exit !(a < b)}'; then t_min=$t; fi
    fi
  done
  if [ -z "$t_min" ]; then echo "  $dev (via $gw): no internet"; continue; fi
  echo "  $dev (via $gw): 2 MB in $t_min s  (one round trip ~ ${rtt} s)"
  if [ -z "$best_t" ] || awk -v a="$t_min" -v b="$best_t" 'BEGIN{exit !(a < b)}'; then
    best_t=$t_min; best_dev=$dev; best_gw=$gw
  fi
done < <(ip -4 route show default | awk '{g=""; d=""; for (i = 1; i < NF; i++) { if ($i == "via") g = $(i+1); if ($i == "dev") d = $(i+1) } if (g != "" && d != "") print g, d}' | sort -u)
if [ -n "$best_dev" ]; then
  ip route add default via "$best_gw" dev "$best_dev" metric 50 initcwnd 40 initrwnd 40 \
    && echo "internet now goes through $best_dev:  $(ip route get 1.1.1.1 | head -1)"
else
  echo "no working internet path found: routes left as they were"
fi
for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > "$p" 2>/dev/null; done
echo "CPU governor: $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null)"
command -v vcgencmd >/dev/null && echo "$(vcgencmd measure_temp)  $(vcgencmd get_throttled)   (throttled=0x0 means the power is fine)"
exit 0
