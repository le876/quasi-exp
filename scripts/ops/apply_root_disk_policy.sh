#!/usr/bin/env bash
set -euo pipefail

# Must run with sudo/root. Applies system-level policies to keep "/" stable.
if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: 请使用 sudo 执行: sudo bash $0"
  exit 1
fi

mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/99-disk-limit.conf <<'EOF'
[Journal]
SystemMaxUse=600M
RuntimeMaxUse=200M
SystemMaxFileSize=64M
RuntimeMaxFileSize=32M
MaxRetentionSec=14day
EOF

cat > /usr/local/sbin/root-disk-housekeeping.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
journalctl --vacuum-size=600M || true
journalctl --vacuum-time=14d || true
if command -v snap >/dev/null 2>&1; then
  snap list --all | awk '/disabled/{print $1, $3}' | while read -r snapname revision; do
    [ -n "${snapname}" ] && [ -n "${revision}" ] && snap remove "${snapname}" --revision="${revision}" || true
  done
  snap set system refresh.retain=2 || true
fi
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true
EOF
chmod 755 /usr/local/sbin/root-disk-housekeeping.sh

cat > /etc/systemd/system/root-disk-housekeeping.service <<'EOF'
[Unit]
Description=Root Disk Housekeeping

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/root-disk-housekeeping.sh
EOF

cat > /etc/systemd/system/root-disk-housekeeping.timer <<'EOF'
[Unit]
Description=Daily Root Disk Housekeeping Timer

[Timer]
OnCalendar=*-*-* 03:30:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl restart systemd-journald
systemctl enable --now root-disk-housekeeping.timer
systemctl start root-disk-housekeeping.service

echo "DONE"
journalctl --disk-usage || true
systemctl --no-pager --full status root-disk-housekeeping.timer | sed -n '1,30p'
