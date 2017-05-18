#include "UdpServerSocket.h"

#define PORT 7080

int main()
{
	UdpServerSocket udp(PORT);
	udp.createSocket();
	udp.bindSocket();

	printf("receive...");
	char* message = udp.receiveMessage();

	printf("%s\n", message);

	return 0;
}