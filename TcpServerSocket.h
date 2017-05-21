#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#include <fstream>
#include <windows.h>
#include <string>
#include <iostream>
#include <dirent.h>
#include "md5.h"
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024 * 30
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
	string getHash(string md5Str); // get hash_value
	char* receiveMessage();
	void sendFile(char* file_name);
};

TcpServerSocket::TcpServerSocket(int port)
{
	this->port = port;
}

void TcpServerSocket::createSocket()
{
	if ((WSAStartup(MAKEWORD(2, 0), &wsaData)) != 0)
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
	Sleep(20);
}

char* TcpServerSocket::receiveMessage()
{
	int mLen = recv(cliSock, buf, BUFSIZE, 0);
	buf[mLen] = 0;
	return buf;
}

string TcpServerSocket::getHash(string md5Str)
{
	md5_state_t state;
	md5_byte_t digest[16];
	char hex_output[16*2+1];

	md5_init(&state);
	md5_append(&state, (const md5_byte_t *)md5Str.c_str(), md5Str.length());
	md5_finish(&state, digest);
	for(int i=0; i<16; i++)
		sprintf(hex_output+i*2, "%02x", digest[i]);

	return hex_output;
}

void TcpServerSocket::sendFile(char* file_name)
{
	string hash_value = "";
	int len;
	FILE *file;


	if( (file = fopen(file_name, "rb")) == NULL)
	{
		perror("fopen : ");
		exit(1);
	}

	while( (len = fread(buf, 1, BUFSIZE, file)) )
	{
		buf[len] = 0;
		hash_value = getHash(hash_value + buf);
		sendMessage(buf);
	}

	// End Of File
	strcpy(buf, "EOF");
	sendMessage(buf);

	//send file size
	fseek(file, 0, SEEK_END);
	int file_size = ftell(file);
	sprintf(buf, "%d", file_size);
	sendMessage(buf);
	fclose(file);

	//send hash_value
	char* hash = (char*)hash_value.c_str();
	sendMessage(hash);
}