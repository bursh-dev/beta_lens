import http.server, ssl, os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "web"))
print("CWD:", os.getcwd())
print("Files:", os.listdir("."))

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cert.pem"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.pem")
)

server = http.server.HTTPServer(("0.0.0.0", 9443), http.server.SimpleHTTPRequestHandler)
server.socket = ctx.wrap_socket(server.socket, server_side=True)

print("Ready: https://192.168.1.212:9443/betalens.html")
server.serve_forever()
