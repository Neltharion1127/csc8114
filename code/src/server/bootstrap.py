import grpc
from concurrent import futures

from proto import fsl_pb2_grpc
from src.shared.common import cfg


def run_server(servicer) -> None:
    max_workers = cfg.get("server", {}).get("max_workers", 10)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    fsl_pb2_grpc.add_FSLServiceServicer_to_server(servicer, server)

    server_port = cfg.get("grpc", {}).get("server_port", 50051)
    server.add_insecure_port(f"[::]:{server_port}")
    server.start()
    print(f"[SERVER] Listening for incoming FSL connections on port {server_port}...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[SERVER] Keyboard interrupt received. Shutting down gracefully...")
    finally:
        print("[SERVER] Executing safety mechanism: Flushing remaining logs...")
        if hasattr(servicer, "flush_logs"):
            servicer.flush_logs()
        server.stop(0)
        print("[SERVER] Shutdown complete.")
