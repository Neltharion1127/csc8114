import grpc
from concurrent import futures
import time

# Import the auto-generated gRPC code
from proto import fsl_pb2
from proto import fsl_pb2_grpc

class FSLServerServicer(fsl_pb2_grpc.FSLServiceServicer):
    """
    A bare-bones implementation of the Server to test if
    it can receive a basic Hello World Ping.
    """
    def Forward(self, request, context):
        # 1. Read the incoming request
        client_id = request.client_id
        data_len = len(request.activation_data)
        
        # 2. Print what we received to prove the network works
        print(f"📦 [SERVER] Received Ping from Client #{client_id}! Payload size: {data_len} bytes.")
        
        # 3. Create a dummy response to send back
        response = fsl_pb2.ForwardResponse(
            gradient_data=b"dummy_gradient_pong",
            status_message="Hello World, PONG from Server!"
        )
        return response

def serve():
    # Start a gRPC server with 10 worker threads
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Attach our custom logic to the server
    fsl_pb2_grpc.add_FSLServiceServicer_to_server(FSLServerServicer(), server)
    
    # Listen on all IP addresses on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    
    print("🚀 [SERVER] Listening for incoming FSL connections on port 50051...")
    
    # Keep the server alive
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
