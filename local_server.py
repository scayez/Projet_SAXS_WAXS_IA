import http.server  # Importing the http.server module for HTTP server functionality
import socketserver  # Importing the socketserver module for socket server functionality
import threading  # Importing threading module for concurrent execution

PORT = 8000  # Port number on which the server will listen
DIRECTORY = "."  # Directory from which the server serves files
httpd = None  # Initialize a global variable to hold the server instance

# Custom request handler class derived from SimpleHTTPRequestHandler
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
        # Initialize the handler with the specified directory

# Function to start the HTTP server
def start_server():
    global httpd
    # Create a TCP server instance with the specified address and handler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    print(f"Serving at port {PORT}")
    httpd.serve_forever()  # Start serving indefinitely

# Function to run the server in a separate thread
def run_server():
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True  # Ensure the server thread exits when the main thread terminates
    server_thread.start()  # Start the server thread


# Example usage:
if __name__ == "__main__":
    run_server()  # Start the HTTP server in a separate thread when this script is run
