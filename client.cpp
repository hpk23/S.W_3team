//#include "UdpClientSocket.h"
#include "TcpClientSocket.h"

#define PORT 7082
#define SERV_PORT 7080

int main()
{
	char* message;
	char IP[100], buf[BUFSIZE], file_name[100];

	strcpy(IP, "127.0.0.1");
	TcpClientSocket tcp(PORT, IP, SERV_PORT);
	tcp.createSocket();
	tcp.connectSocket();

	strcpy(buf, "I want to download a file");
	tcp.sendMessage(buf);

	strcpy(buf, tcp.receiveMessage());
	strcpy(file_name, buf);

	tcp.receiveFile(file_name);

	return 0;

	/*char IP[100], buf[BUFSIZE], file_name[100];
	IN_ADDR addr;

	strcpy(IP, "127.0.0.1");
	UdpClientSocket udp(PORT, IP, SERV_PORT);
	udp.createSocket();

	strcpy(buf, "I want to download a file");
	udp.sendMessage(buf);

	strcpy(buf, udp.receiveMessage());
	strcpy(file_name, buf);

	udp.receiveFile(file_name);

	return 0;*/
}