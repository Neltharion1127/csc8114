import os
import threading
from datetime import datetime

import grpc
import torch
from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.models.split_lstm import ServerHead
from src.server.bootstrap import run_server
from src.server.fedavg import FedAvgCoordinator
from src.server.forward_service import handle_forward_request
from src.server.reporting import ServerReporter
from src.server.scheduler import CompressionScheduler

from src.shared.common import cfg, project_root
from src.shared.serialization import bytes_to_tensor

class FSLServerServicer(fsl_pb2_grpc.FSLServiceServicer):
    """gRPC servicer for federated split learning."""

    def __init__(self):
        self.hidden_size = cfg.get("model", {}).get("hidden_size", 64)
        lr = cfg.get("training", {}).get("lr", 0.001)
        use_gpu = cfg.get("training", {}).get("use_gpu", False)
        self.device = torch.device("mps") if (use_gpu and torch.backends.mps.is_available()) else torch.device("cpu")
        print(f"[SERVER] Using device: {self.device} (use_gpu={use_gpu})")

        self.server_model = ServerHead(hidden_size=self.hidden_size, output_size=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.server_model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.num_clients = cfg.get("federated", {}).get("num_clients", 3)
        self.sync_lock = threading.Lock()

        self._next_client_id = 1
        self._reg_lock = threading.Lock()
        self._client_name_to_id: dict[str, int] = {}
        self._assigned_ids: set[int] = set()

        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = os.path.join(project_root, "bestweights", self.session_id)
        self.periodic_dir = os.path.join(self.session_dir, "periodic")
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.periodic_dir, exist_ok=True)
        self.ckpt_interval = cfg.get("training", {}).get("checkpoint_interval", 10)
        print(f"[SERVER] Session ID: {self.session_id}  →  {self.session_dir}")
        print(f"[SERVER] Periodic checkpoint every {self.ckpt_interval} rounds  →  {self.periodic_dir}")
        self.fedavg = FedAvgCoordinator(
            num_clients=self.num_clients,
            hidden_size=self.hidden_size,
            session_id=self.session_id,
            session_dir=self.session_dir,
            periodic_dir=self.periodic_dir,
            ckpt_interval=self.ckpt_interval,
        )

        scheduler_cfg = cfg.get("scheduler", {})
        self.scheduler = CompressionScheduler(
            default_mode=cfg.get("compression", {}).get("mode", "float32"),
            enabled=scheduler_cfg.get("enabled", True),
            float16_threshold=scheduler_cfg.get("latency_threshold", 4.0),
            int8_threshold=scheduler_cfg.get("int8_latency_threshold", 10.0),
        )
        self.log_server_requests = cfg.get("console", {}).get("log_server_requests", False)
        self.profiler_enabled = cfg.get("profiler", {}).get("enabled", True)
        self.scheduler_enabled = scheduler_cfg.get("enabled", True)
        self.reporter = ServerReporter(session_id=self.session_id)

    def Register(self, request, context):
        """Assign a client id and return the shared session id."""
        with self._reg_lock:
            client_name = request.client_name or f"client-{self._next_client_id}"
            requested_id = request.requested_client_id

            if client_name in self._client_name_to_id:
                assigned_id = self._client_name_to_id[client_name]
            elif requested_id > 0 and requested_id not in self._assigned_ids:
                assigned_id = requested_id
                self._client_name_to_id[client_name] = assigned_id
                self._assigned_ids.add(assigned_id)
            else:
                while self._next_client_id in self._assigned_ids:
                    self._next_client_id += 1
                assigned_id = self._next_client_id
                self._client_name_to_id[client_name] = assigned_id
                self._assigned_ids.add(assigned_id)
                self._next_client_id += 1

            if assigned_id >= self._next_client_id:
                self._next_client_id = assigned_id + 1

        print(
            f"[SERVER] Client registered — name: {client_name} | requested_id: {requested_id or 'auto'} "
            f"| assigned_id: {assigned_id} | session: {self.session_id}"
        )
        return fsl_pb2.RegisterResponse(
            client_id=assigned_id,
            total_clients=self.num_clients,
            session_id=self.session_id,
        )

    def Forward(self, request, context):
        """Handle one forward request from a client."""
        try:
            client_id = getattr(request, "client_id", -1)
            reported_latency = getattr(request, "latency_ms", 0.0)
            assigned_compression = self.scheduler.assign(client_id, reported_latency)
            result = handle_forward_request(
                request,
                hidden_size=self.hidden_size,
                device=self.device,
                server_model=self.server_model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                sync_lock=self.sync_lock,
                current_round=self.fedavg.current_round,
                assigned_compression=assigned_compression,
                profiler_enabled=self.profiler_enabled,
                scheduler_enabled=self.scheduler_enabled,
            )
            self.reporter.record(result.log_entry)
            if self.log_server_requests:
                print(result.monitor_message)

            return result.response

        except Exception as e:
            print(f"[SERVER ERROR] Processing failed: {str(e)}")
            return fsl_pb2.ForwardResponse(
                status_message=f"Error: {str(e)}",
                success=False,
            )

    def flush_logs(self):
        self.reporter.flush()

    def Synchronize(self, request, context):
        """Aggregate client weights and return the latest global model."""
        try:
            local_weights = bytes_to_tensor(request.client_weights)
            return self.fedavg.synchronize(
                request,
                local_weights=local_weights,
                server_model=self.server_model,
                optimizer=self.optimizer,
            )
        except Exception as e:
            print(f"[FED AVG ERROR] Synchronization failed: {str(e)}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return fsl_pb2.SyncResponse()


def serve():
    run_server(FSLServerServicer())


if __name__ == "__main__":
    serve()
