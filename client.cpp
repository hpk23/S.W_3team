#include "UdpClientSocket.h"

#define PORT 7082
#define SERV_PORT 7080

int main()
{
	char IP[100];
	IN_ADDR addr;

	strcpy(IP, "127.0.0.1");
	UdpClientSocket udp(PORT, IP, SERV_PORT);
	udp.createSocket();

	char buf[BUFSIZE];

	strcpy(buf, "hello");

	udp.sendMessage(buf);

	return 0;
}