import socket
HOST = '127.0.0.1' # The server’s hostname or IP address
PORT = 65432 # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    # s.sendall(b'Hello, world')
    data = s.recv(1024)
    print('Received from server:', data.decode())
    # print('Received', repr(data))
#     print(data)

    while(1):
        msg=input('Enter message to send:').encode()
        try:
            # Set the whole string
            s.sendto(msg, (HOST, PORT))
            # receive data from client (data, addr)
            d= s.recvfrom(1024)
            reply = d[0].decode()
            addr = d[1]
            print ('Server reply : ' + reply)
        except socket.error as msg:
            print ('Error Code: ' + str(msg[0]) + ' Message ' + msg[1])
