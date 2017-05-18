#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#include <fstream>

#define BUFSIZE 1024*24
#pragma comment(lib, "ws2_32.lib")

using namespace std;


class UdpServerSocket
{
private :
	int port;
	int cliLen; // Length of incoming message
	
	WSADATA wsaData;
	
	struct sockaddr_in servAddr;
	struct sockaddr_in cliAddr;

	SOCKET servSock;
	SOCKET cliSock;
	
	char buf[BUFSIZE+1];
public :
	UdpServerSocket(int port);
	void createSocket();
	void bindSocket();
	void sendMessage(char* message);
	void sendFile(char* file_name);
	char* receiveMessage();
};

UdpServerSocket::UdpServerSocket(int port)
{
	this->port = port;
}

void UdpServerSocket::createSocket()
{
	if ((WSAStartup(MAKEWORD(2, 2), &wsaData)) != 0)
	{
		perror("WSA :");
		exit(1);
	}

	if ((servSock = socket(PF_INET, SOCK_DGRAM, 0)) == INVALID_SOCKET)
	{
		perror("servSock :");
		exit(1);
	}

	memset(&servAddr, 0, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	servAddr.sin_port = htons(port);

}

void UdpServerSocket::bindSocket()
{
	if (bind(servSock, (SOCKADDR *)&servAddr, sizeof(servAddr)) == SOCKET_ERROR)
	{
		perror("bind error : ");
		exit(1);
	}

	cliLen = sizeof(cliAddr);
}

char* UdpServerSocket::receiveMessage()
{
	int mLen = recvfrom(servSock, buf, BUFSIZE, 0, (SOCKADDR *)&cliAddr, &cliLen);
	buf[mLen] = 0;
	return buf;
}

void UdpServerSocket::sendMessage(char* message)
{
	int mLen = strlen(message);
	strcpy(buf, message);
	sendto(servSock, buf, mLen, 0, (struct sockaddr *)&cliAddr, sizeof(cliAddr));
}

void UdpServerSocket::sendFile(char* file_name)
{
	ifstream file(file_name);

	if(file.is_open())
	{
		while(file.getline(buf, BUFSIZE))
		{
			strcat(buf, "\n");
			sendMessage(buf);
		}
		file.close();
	}
	strcpy(buf, "EOF");
	sendMessage(buf);
}