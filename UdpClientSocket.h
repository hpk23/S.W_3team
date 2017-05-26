#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#include <fstream>
#include <windows.h>
#include <iostream>
#include <dirent.h>
#include <windows.h>
#include <time.h>
#include "md5.h"
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024 * 10

using namespace std;

class UdpClientSocket
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
	UdpClientSocket(int port, char* ip, int serv_port);
	void createSocket();
	void sendMessage(char* message);
	void receiveFile(char* file_name);
	char* receiveMessage();
	string getHash(string md5Str);
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
	Sleep(20);
}

char* UdpClientSocket::receiveMessage()
{
	int mLen = recvfrom(sock, buf, BUFSIZE, 0, (SOCKADDR *)&peerAddr, &peerLen);

	if(memcmp(&peerAddr, &servAddr, sizeof(peerAddr)))
	{
		printf("It is not a file from the server.\n");
		exit(1);
	}
	buf[mLen] = 0;
	return buf;
}

string UdpClientSocket::getHash(string md5Str)
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

void UdpClientSocket::receiveFile(char* file_name)
{
	printf("UDP protocol\n\n");

	char directory[1024] = "";
	int f_len = strlen(file_name), idx = 0;
	double receive_size = 0.0;
	char temp [1024];

	for(int i=0; i<f_len; i++)
	{
		if(file_name[i] == '/')
		{
			temp[idx] = 0;
			strcat(directory, temp);
			idx = 0;
		}
		else temp[idx++] = file_name[i];
	}
	
	FILE* outFile;

	if((outFile = fopen(file_name, "wb")) == NULL)
	{
		if(CreateDirectory(directory, NULL)) {}
		if((outFile = fopen(file_name, "wb")) == NULL)
		{
			perror("outFile open : ");
			exit(1);
		}
	}

	strcpy(buf, receiveMessage());

	clock_t start_time = clock();
	int cnt = 0;
	while(strcmp(buf, "EOF"))
	{
		int size = strlen(buf);
		fwrite(buf, sizeof(buf[0]), size, outFile);
		strcpy(buf, receiveMessage());
		receive_size += (double)size;
		if(cnt % 50 == 0) printf("%.2fMB/sec\n", (receive_size/(1024.0*1024.0)) / ((double)(clock() - start_time) / CLOCKS_PER_SEC));
		cnt++;
	}
	printf("%.2fMB/sec\n", (receive_size/(1024.0*1024.0)) / ((double)(clock() - start_time) / CLOCKS_PER_SEC));
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
		printf("The file was successfully received.\n");
	else
	{
		printf("The file download failed... Please try again\n");
		exit(1);
	}
}