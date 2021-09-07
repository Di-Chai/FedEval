from enum import Enum

class ServerFlaskInterface(Enum):
    Dashboard = '/dashboard'
    Status = '/status'
    DownloadPattern = '/download/{}'
