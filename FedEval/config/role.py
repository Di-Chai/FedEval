from enum import Enum


class Role(Enum):
    Server = 'server'
    Client = 'client'


ClientId = int      # to identify client
