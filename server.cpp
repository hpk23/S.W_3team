//#include "UdpServerSocket.h"
#include "TcpServerSocket.h"
#define PORT 7080

int main()
{
	char* message;
	char buf[BUFSIZE];

	TcpServerSocket tcp(PORT);
	tcp.createSocket();
	tcp.bindSocket();
	tcp.listenSocket();
	tcp.acceptSocket();

	message = tcp.receiveMessage();
	printf("%s\n", message);

	strcpy(buf, "client/step3.txt");
	tcp.sendMessage(buf);

	strcpy(buf, "D:/S.W_3team/step3.txt");
	tcp.sendFile(buf);

	return 0;
}
