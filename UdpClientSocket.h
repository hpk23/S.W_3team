#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024*24

class UdpClientSocket
{
private :
	char servIp[50];
	char buf[BUFSIZE + 1];
	
	int port;
	int servPort;
	
	WSADATA wsaData;

	sockaddr_in servAddr;

	SOCKET sock;

public :
	UdpClientSocket(int port, char* ip, int serv_port);
	void createSocket();
	void sendMessage(char* message);
};

UdpClientSocket::UdpClientSocket(int port, char* ip, int servPort)
{
	this->port = port;
	this->servPort = servPort;
	strcpy(servIp, ip);
}

void UdpClientSocket::createSocket()
{
	if ((WSAStartup(MAKEWORD(2, 2), &wsaData)) != 0)
	{
		perror("WSA : ");
		exit(1);
	}

	if ((sock = socket(PF_INET, SOCK_DGRAM, 0)) == INVALID_SOCKET)
	{
		perror("sock : ");
		exit(1);
	}

	memset(&servAddr, 0, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_addr.s_addr = inet_addr(servIp);
	servAddr.sin_port = htons(servPort);

	//connect(sock, (SOCKADDR *)&servAddr, sizeof(servAddr));
}

void UdpClientSocket::sendMessage(char* message)
{
	int mLen = strlen(message);
	strcpy(buf, message);
	sendto(sock, buf, mLen, 0, (struct sockaddr *)&servAddr, sizeof(servAddr));
}