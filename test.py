import socket

string = "等一下啊|2.1|4|4|0.5"
address = ("106.75.224.197", 2222)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(address)
s.send(bytes(string, "utf-8"))
data_hook = s.recv(1024)
s.close()
print(data_hook.decode("utf-8"))
