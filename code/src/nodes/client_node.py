import grpc
import time

# Import the auto-generated gRPC code
from proto import fsl_pb2
from proto import fsl_pb2_grpc

def run_client():
    print("⏳ [CLIENT] Waiting 3 seconds for Server to boot up...")
    time.sleep(3)
    print("🚀 [CLIENT] Connecting to Server....")
    
    # Establish connection with the Docker Compose hostname "fsl-server"
    with grpc.insecure_channel('fsl-server:50051') as channel:
        
        # Create a stub (the local representative of the remote server)
        stub = fsl_pb2_grpc.FSLServiceStub(channel)
        
        try:
            # Create a mock Hello World message matching our proto structure
            request = fsl_pb2.ForwardRequest(
                client_id=1,
                activation_data=b"dummy_activation_ping",
                true_target=0.0,
                latency_ms=15.2
            )
            
            # Send the request over the network bridge and wait for the response
            print("📤 [CLIENT] Sending Ping Request...")
            response = stub.Forward(request)
            
            # Print the response we got from the server
            print(f"📥 [CLIENT] Response Received! Server said: '{response.status_message}'")
            print(f"📥 [CLIENT] Gradient bytes returned: {len(response.gradient_data)} bytes")
            print("✅ [SUCCESS] Network bridge works perfectly!")
            
        except grpc.RpcError as e:
            print(f"❌ [CLIENT ERROR] gRPC failed: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run_client()
