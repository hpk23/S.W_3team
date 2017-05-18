#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024*24

using namespace std;

class UdpClientSocket
{
private :
	char servIp[50];
	char buf[BUFSIZE + 1];
	
	int port;
	int servPort;
	int peerLen;
	
	WSADATA wsaData;

	sockaddr_in servAddr;
	sockaddr_in peerAddr;

	SOCKET sock;

public :
	UdpClientSocket(int port, char* ip, int serv_port);
	void createSocket();
	void sendMessage(char* message);
	void receiveFile(char* file_name);
	char* receiveMessage();
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

	peerLen = sizeof(peerAddr);
}

void UdpClientSocket::sendMessage(char* message)
{
	int mLen = strlen(message);
	strcpy(buf, message);
	sendto(sock, buf, mLen, 0, (struct sockaddr *)&servAddr, sizeof(servAddr));
}

char* UdpClientSocket::receiveMessage()
{
	int mLen = recvfrom(sock, buf, BUFSIZE, 0, (SOCKADDR *)&peerAddr, &peerLen);

	if(memcmp(&peerAddr, &servAddr, sizeof(peerAddr)))
	{
		printf("It is not data from the server..\n");
		exit(1);
	}
	buf[mLen] = 0;
	return buf;
}

void UdpClientSocket::receiveFile(char* file_name)
{
	ofstream outFile(file_name);

	strcpy(buf, receiveMessage());
	while(strcmp(buf, "EOF"))
	{
		if(outFile.is_open())
		{
			outFile << buf;
			printf("%s", buf);
			strcpy(buf, receiveMessage());
		}
		else
		{
			perror("outFile : ");
			exit(1);
		}
	}
	outFile.close();
}