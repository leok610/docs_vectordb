import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
from pathlib import Path

# Add the src directory to sys.path so we can import our server
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root / "src"))

from docs_vectordb.embedding_server import app
from waitress import serve

class EmbeddingService(win32serviceutil.ServiceFramework):
    _svc_name_ = "DocsEmbeddingService"
    _svc_display_name_ = "Docs VectorDB Embedding Service"
    _svc_description_ = "Provides a persistent local API for generating SentenceTransformer embeddings."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        
        # Run the Waitress server in the background
        # Note: In a real service, you might want more robust thread management
        # but for this utility, this is sufficient.
        serve(app, host="127.0.0.1", port=5000)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(EmbeddingService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(EmbeddingService)
