import csv
from datetime import datetime
from zoneinfo import ZoneInfo

from scapy.all import *

attack_machines = ["125.99.164.116"]
target_machines = ["35.154.126.159", "172.31.2.36"]

fields = ['TIMESTAMP', 'SRC', 'DST', 'DIRECTION', 'IP_TOS', 'DF', 'MF', 'IP_TTL', 'TCP_SPORT', 'TCP_DPORT', 'TCP_SEQ',
          'TCP_ACK', 'TCP_RESERVED', 'FPA', 'FA', 'A', 'S', 'SA', 'PA', 'R', 'RA', 'TCP_WINDOW', 'TCP_MSS', 'TCP_NOP',
          'TCP_WSCALE', 'TYPE']
file = open("packets.csv", 'a')
csv_writer = csv.writer(file)
csv_writer.writerow(fields)

tcp_flags = ['FPA', 'FA', 'A', 'S', 'SA', 'PA', 'R', 'RA']


def sniff_packets():
    pkts = sniff(timeout=600, filter="tcp and ip")
    print(pkts)
    for packet in pkts:
        try:
            tcp = packet[TCP]
            ip = packet[IP]
        except:
            continue

        if tcp.dport == "ssh" or tcp.sport == "ssh": continue
        if tcp.dport == 22 or tcp.sport == 22: continue

        timestamp = datetime.now(tz=ZoneInfo('Asia/Kolkata'))

        row = []
        row.append(str(timestamp))

        src = ip.src.replace(".", "")
        dst = ip.dst.replace(".", "")

        direction = 0
        if ip.src in target_machines:
            direction = 1

        row += [src, dst, direction, ip.tos]
        if ip.flags == "DF":
            row += ["1", "0"]
        else:
            row += ["0", "1"]

        row.append(ip.ttl)

        # TCP
        temp = ["0" for x in range(len(tcp_flags))]
        temp[tcp_flags.index(tcp.flags)] = "1"

        row += [tcp.sport, tcp.dport, tcp.seq, tcp.ack, tcp.reserved] + temp + [tcp.window]

        mss = ""
        sackok = ""
        nop = ""
        wscale = ""
        for key, val in tcp.options:
            if key == 'MSS':
                mss = val
            elif key == 'SAckOK':
                sackok = str(val).replace("'", " ")
            elif key == 'NOP':
                if val: nop = val
            elif key == 'WScale':
                wscale = val

        row += [mss, nop, wscale]

        # 0 = benign, 1 = attack
        if ip.src in attack_machines:
            row.append(1)
        else:
            row.append(0)

        temp = [str(x) for x in row]
        # print(",".join(temp))
        csv_writer.writerow(row)


while True:
    f = open("in", "r")
    line = f.readline()
    if line == 'STOP\n': break
    sniff_packets()
