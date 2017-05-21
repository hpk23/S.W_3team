#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#include <fstream>
#include <windows.h>
#include <iostream>
#include <dirent.h>
#include "md5.h"
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024 * 30

using namespace std;

class TcpClientSocket
{
private :
	char servIp[50];
	char buf[BUFSIZE + 5];
	
	int port;
	int servPort;
	int peerLen;
	
	WSADATA wsaData;

	sockaddr_in servAddr;
	sockaddr_in peerAddr;

	SOCKET sock;

public :
	TcpClientSocket(int port, char* ip, int serv_port);
	void createSocket();
	void connectSocket();
	void sendMessage(char* message);
	char* receiveMessage();
	void receiveFile(char* file_name);
	string getHash(string md5Str);
};

TcpClientSocket::TcpClientSocket(int port, char* ip, int serv_port)
{
	this->port = port;
	strcpy(servIp, ip);
	servPort = serv_port;
}

void TcpClientSocket::createSocket()
{
	if ((WSAStartup(MAKEWORD(2, 0), &wsaData)) != 0)
	{
		perror("WSA : ");
		exit(1);
	}

	if ((sock = socket(PF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
	{
		perror("sock : ");
		exit(1);
	}

	memset(&servAddr, 0, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_addr.s_addr = inet_addr(servIp);
	servAddr.sin_port = htons(servPort);
}

void TcpClientSocket::connectSocket()
{
	if( (connect(sock, (struct sockaddr *)&servAddr, sizeof(servAddr))) == INVALID_SOCKET)
	{
		perror("connect : ");
		exit(1);
	}
}

char* TcpClientSocket::receiveMessage()
{
	int mLen = recv(sock, buf, BUFSIZE, 0);

	buf[mLen] = 0;
	return buf;
}

void TcpClientSocket::sendMessage(char* message)
{
	strcpy(buf, message);
	send(sock, buf, strlen(buf), 0);
	Sleep(20);
}

string TcpClientSocket::getHash(string md5Str)
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

void TcpClientSocket::receiveFile(char* file_name)
{
	FILE* outFile;

	if((outFile = fopen(file_name, "wb")) == NULL)
	{
		perror("outFile open : ");
		exit(1);
	}

	strcpy(buf, receiveMessage());
	while(strcmp(buf, "EOF"))
	{
		fwrite(buf, sizeof(buf[0]), strlen(buf), outFile);
		strcpy(buf, receiveMessage());
	}
	fclose(outFile);

	string receive_hash_value;

	// receive file size
	string receive_file_size = receiveMessage();

	// receive hash value
	receive_hash_value = receiveMessage();

	// get my hash value
	FILE* myFile;
	int len;

	string my_hash_value = "";

	if( (myFile = fopen(file_name, "rb")) == NULL )
	{
		perror("myfile open : ");
		exit(1);
	}

	while( (len = fread(buf, 1, BUFSIZE, myFile)) )
	{
		buf[len] = 0;
		my_hash_value = getHash(my_hash_value + buf);
	}

	//get my file size
	fseek(myFile, 0, SEEK_END);
	int my_file_size = ftell(myFile);
	sprintf(buf, "%d", my_file_size);

	printf("receive_file_size : %s, my_file_size : %s\n", receive_file_size.c_str(), buf);
	fclose(myFile);


	//compare receive_hash_value, my_hash_value
	if(receive_hash_value == my_hash_value)
		printf("The file was successfully received.");
	else
		printf("The file download failed... Please try again");
}