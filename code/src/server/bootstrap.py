import grpc
from concurrent import futures
import os

from proto import fsl_pb2_grpc
from src.shared.common import cfg
from src.shared.runtime import grpc_channel_options


def run_server(servicer) -> None:
    max_workers = cfg.get("server", {}).get("max_workers", 10)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=grpc_channel_options(),
    )
    fsl_pb2_grpc.add_FSLServiceServicer_to_server(servicer, server)

    grpc_cfg = cfg.get("grpc", {})
    server_port = grpc_cfg.get("server_port", 50051)
    bind_host = os.getenv("FSL_SERVER_BIND_HOST", str(grpc_cfg.get("bind_host", "[::]")))
    bind_addr = f"{bind_host}:{server_port}"
    tls_enabled = grpc_cfg.get("tls_enabled", False)

    if tls_enabled:
        cert_path = grpc_cfg.get("tls_cert_path")
        key_path = grpc_cfg.get("tls_key_path")
        if not cert_path or not key_path:
            raise RuntimeError(
                "grpc.tls_enabled is true but tls_cert_path / tls_key_path are not set in config."
            )
        with open(cert_path, "rb") as fh:
            cert_pem = fh.read()
        with open(key_path, "rb") as fh:
            key_pem = fh.read()
        credentials = grpc.ssl_server_credentials([(key_pem, cert_pem)])
        bind_result = server.add_secure_port(bind_addr, credentials)
        print(f"[SERVER] TLS enabled (cert={cert_path})")
    else:
        bind_result = server.add_insecure_port(bind_addr)
        if bind_result == 0 and bind_host == "[::]":
            fallback_addr = f"0.0.0.0:{server_port}"
            print(f"[SERVER] Failed to bind {bind_addr}; retrying with {fallback_addr}")
            bind_result = server.add_insecure_port(fallback_addr)
            bind_addr = fallback_addr

    if bind_result == 0:
        raise RuntimeError(
            f"Failed to bind gRPC server on {bind_addr}. "
            "Try setting FSL_SERVER_BIND_HOST=0.0.0.0."
        )
    server.start()
    print(f"[SERVER] Listening for incoming FSL connections on {bind_addr}...")

    try:
        while True:
            # Timeout is only used as a periodic tick for checking custom shutdown conditions.
            # Do not branch on the return value here because grpc's return semantics are not
            # a reliable "server terminated" signal across versions.
            server.wait_for_termination(timeout=1.0)
            if hasattr(servicer, "should_shutdown") and servicer.should_shutdown():
                print("[SERVER] Shutdown requested by servicer. Stopping server...")
                break
    except KeyboardInterrupt:
        print("\n[SERVER] Keyboard interrupt received. Shutting down gracefully...")
    finally:
        print("[SERVER] Executing safety mechanism: Flushing remaining logs...")
        if hasattr(servicer, "flush_logs"):
            servicer.flush_logs()
        # Give a 1s grace period for pending RPC threads to finish to avoid "terminate called" crash.
        server.stop(1).wait()
        print("[SERVER] Shutdown complete.")
