#include "UdpClientSocket.h"
#include "TcpClientSocket.h"

#define PORT 7082
#define UDP_SERV_PORT 7709
#define TCP_SERV_PORT 7710

void menu();

int main()
{
	char* message;
	char IP[100], buf[BUFSIZE], file_name[100];
	int select;

	strcpy(IP, "127.0.0.1");

	UdpClientSocket udp(PORT, IP, UDP_SERV_PORT);
	TcpClientSocket tcp(PORT, IP, TCP_SERV_PORT);

	udp.createSocket();
	tcp.createSocket();

	menu();

	while(true)
	{
		scanf("%d", &select);
		if(select == 1 || select == 2) break;
		else printf("Invalid number\n\n");
	}

	// send select number
	sprintf(buf, "%d", select);
	udp.sendMessage(buf);

	Sleep(1000);

	// UDP
	if(select == 1)
	{
		// receive data list
		message = udp.receiveMessage();

		while(true)
		{
			printf("%s\n", message);
			printf("Please select a number : ");
			scanf("%d", &select);
			if(select == 1 || select == 2 || select == 3) break;
			else printf("Invalid number\n\n");
		}

		// send select number
		sprintf(buf, "%d", select);
		udp.sendMessage(buf);

		// receive file or directory
		while(true)
		{
			strcpy(buf, udp.receiveMessage());

			if(!strcmp(buf, "CLEAR")) break;

			strcpy(file_name, buf);
			udp.receiveFile(file_name);
		}
	}

	// TCP
	else if(select == 2)
	{
		tcp.connectSocket();
		// receive data list
		message = tcp.receiveMessage();
		printf("%s\n", message);
		printf("Please select a number : ");
		scanf("%d", &select);

		// send select number
		sprintf(buf, "%d", select);
		tcp.sendMessage(buf);

		// receive file or directory 
		while(true)
		{
			strcpy(buf, tcp.receiveMessage());
			
			if(!strcmp(buf, "CLEAR")) break;
			
			strcpy(file_name, buf);
			printf("file_name : %s\n", file_name);
			tcp.receiveFile(file_name);
		}
	}

	return 0;

	/*strcpy(IP, "127.0.0.1");
	TcpClientSocket tcp(PORT, IP, SERV_PORT);
	tcp.createSocket();
	tcp.connectSocket();

	strcpy(buf, "I want to download a file");
	tcp.sendMessage(buf);

	strcpy(buf, tcp.receiveMessage());
	strcpy(file_name, buf);

	tcp.receiveFile(file_name);

	return 0;*/

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

void menu()
{
	printf("1. UDP\n");
	printf("2. TCP\n");
	printf("Please select a number : ");
}