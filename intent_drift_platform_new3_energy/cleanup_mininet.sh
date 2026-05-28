#!/bin/bash
echo "=== 彻底清理 Mininet ==="
sudo mn -c 2>/dev/null
sudo killall -9 ovs-testcontroller controller ovsdb-server ovs-vswitchd 2>/dev/null
sudo pkill -9 -f "ryu-manager" 2>/dev/null
for br in $(sudo ovs-vsctl list-br 2>/dev/null); do
    echo "Deleting bridge: $br"
    sudo ovs-vsctl --if-exists del-br $br
done
sudo ip -all netns delete 2>/dev/null
for intf in $(ip link show | grep -oE "(s[0-9]+-eth[0-9]+|h[0-9]+-eth[0-9]+)" | sort -u); do
    echo "Deleting interface: $intf"
    sudo ip link delete $intf 2>/dev/null
done
sudo service openvswitch-switch restart
sleep 2
sudo mn -c 2>/dev/null
echo "=== 清理完成 ==="
