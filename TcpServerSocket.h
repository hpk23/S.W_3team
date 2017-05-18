#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#include <fstream>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024
#define LISTENNUM 5

using namespace std;

class TcpServerSocket
{
private :
	int port;
	int cliLen; // Length of incoming message
	
	WSADATA wsaData;
	
	struct sockaddr_in servAddr;
	struct sockaddr_in cliAddr;

	SOCKET servSock;
	SOCKET cliSock;
	
	char buf[BUFSIZE+5];
public :
	TcpServerSocket(int port);
	void createSocket();
	void bindSocket();
	void listenSocket();
	void acceptSocket();
	void sendMessage(char* message);
	char* receiveMessage();
	void sendFile(char* file_name);
};

TcpServerSocket::TcpServerSocket(int port)
{
	this->port = port;
}

void TcpServerSocket::createSocket()
{
	if ((WSAStartup(MAKEWORD(2, 2), &wsaData)) != 0)
	{
		perror("WSA :");
		exit(1);
	}

	if ((servSock = socket(PF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
	{
		perror("servSock :");
		exit(1);
	}

	memset(&servAddr, 0, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	servAddr.sin_port = htons(port);
}

void TcpServerSocket::bindSocket()
{
	if (bind(servSock, (SOCKADDR *)&servAddr, sizeof(servAddr)) == SOCKET_ERROR)
	{
		perror("bind error : ");
		exit(1);
	}
}

void TcpServerSocket::listenSocket()
{
	if(listen(servSock, LISTENNUM) == SOCKET_ERROR)
	{
		perror("listen error : ");
		exit(1);
	}
}

void TcpServerSocket::acceptSocket()
{
	cliLen = sizeof(cliAddr);
	
	cliSock=accept(servSock, (struct sockaddr *)&cliAddr, &cliLen);
	if(cliSock==INVALID_SOCKET)
	{
		perror("accept error : ");
		exit(1);
	}
}

void TcpServerSocket::sendMessage(char* message)
{
	strcpy(buf, message);
	send(cliSock, buf, strlen(buf), 0);
	Sleep(100);
}

char* TcpServerSocket::receiveMessage()
{
	int mLen = recv(cliSock, buf, BUFSIZE, 0);

	buf[mLen] = 0;
	return buf;
}

void TcpServerSocket::sendFile(char* file_name)
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