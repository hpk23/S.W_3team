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


class UdpServerSocket
{
private :
	int port;
	int cliLen; // Length of incoming message
	
	WSADATA wsaData;
	
	struct sockaddr_in servAddr;
	struct sockaddr_in cliAddr;

	SOCKET servSock;
	
	char buf[BUFSIZE+5];
public :
	UdpServerSocket(int port);
	void createSocket();
	void bindSocket();
	void sendMessage(char* message);
	string getHash(string md5Str);
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
	Sleep(10);
}

string UdpServerSocket::getHash(string md5Str)
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

void UdpServerSocket::sendFile(char* file_name)
{
	printf("UDP protocol\n\n");
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