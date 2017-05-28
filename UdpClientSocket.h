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
#include <io.h>
#include <vector>
#include "md5.h"
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 10240

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

	vector<string> corrupted_file_list;

public :
	UdpClientSocket(int port, char* ip, int serv_port);
	void createSocket();
	void closeSocket();
	void sendMessage(char* message);
	void receiveFile(string f_name);
	void resumeFile(string f_name);
	char* receiveMessage();
	string getHash(string md5Str);
	getFileSize(char* file_name);
	void clearCorruptedFileList();
	vector<string> getCorruptedFileList();
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

void UdpClientSocket::closeSocket()
{
	closesocket(sock);
}

void UdpClientSocket::sendMessage(char* message)
{
	int mLen = strlen(message);
	strcpy(buf, message);
	sendto(sock, buf, mLen, 0, (struct sockaddr *)&servAddr, sizeof(servAddr));
	Sleep(25);
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

int UdpClientSocket::getFileSize(char* file_name)
{
	FILE *file;

	if( (file = fopen(file_name, "rb")) == NULL)
	{
		perror("fopen : ");
		exit(1);
	}

	fseek(file, 0, SEEK_END);
	int file_size = ftell(file);

	return file_size;
}

void UdpClientSocket::resumeFile(string f_name)
{
	char* file_name = (char*)f_name.c_str();
	int len;

	FILE* file;

	if((file = fopen(file_name, "rb+")) == NULL)
	{
		perror("file : ");
		exit(1);
	}


	// send exist file size
	int exist_file_size = getFileSize(file_name);
	sprintf(buf, "%d", exist_file_size);
	sendMessage(buf);


	// receive exist file state
	strcpy(buf, receiveMessage());
	if(!strcmp(buf, "strange file"))
	{
		printf("%s file is corrupt. Please delete it and try again\n", file_name);
		corrupted_file_list.push_back(file_name);
		return;
	}
	else if(!strcmp(buf, "same"))
	{
		printf("file download already done\n");
		return;
	}
	else printf("%s file is ok. start download\n", file_name);

	// get exist file hash value
	string exist_file_hash_value = "";
	while( (len = fread(buf, 1, BUFSIZE, file)) )
	{
		buf[len] = 0;
		exist_file_hash_value = getHash(exist_file_hash_value + buf);
	}

	// receive server file hash value
	string server_file_hash_value = receiveMessage();

	// send file state
	if(exist_file_hash_value != server_file_hash_value)
	{
		printf("%s file is corrupt. Please delete it and try again\n", file_name);
		strcpy(buf, "corrupt");
		sendMessage(buf);
		corrupted_file_list.push_back(file_name);
		Sleep(500);
		return;
	}
	else
	{
		strcpy(buf, "OK");
		sendMessage(buf);
		Sleep(500);
	}

	// receive file
	strcpy(buf, receiveMessage());
	int N = atoi(buf);
	clock_t start_time = clock();
	int cnt = 1;
	double receive_size = 0.0;
	exist_file_hash_value = "";
	for(int i=0; i<N; i++)
	{
		strcpy(buf, receiveMessage());
		int size = strlen(buf);
		fwrite(buf, sizeof(buf[0]), size, file);
		exist_file_hash_value = getHash(exist_file_hash_value + buf);
		receive_size += (double)size;
		if(cnt % 100 == 0) printf("%.2fMB/sec\n", (receive_size/(1024.0*1024.0)) / ((double)(clock() - start_time) / CLOCKS_PER_SEC));
		cnt++;
		Sleep(20);
	}
	if(N != 0)
		printf("%.2fMB/sec\n", (receive_size/(1024.0*1024.0)) / ((double)(clock() - start_time) / CLOCKS_PER_SEC));
	else printf("0.0MB/sec\n");
	fclose(file);

	// receive server file hash value
	server_file_hash_value = receiveMessage();

	// receive server file size
	strcpy(buf, receiveMessage());
	int receive_file_size = atoi(buf);

	int my_file_size = getFileSize(file_name);

	//compare receive_hash_value, my_hash_value
	if(server_file_hash_value == exist_file_hash_value && receive_file_size == my_file_size)
		printf("The file was successfully received.\n");
	else
	{
		printf("%s file download failed... Please try again\n", file_name);
		corrupted_file_list.push_back(file_name);
	}

}

void UdpClientSocket::receiveFile(string f_name)
{
	int len;
	char* file_name = (char*)f_name.c_str();
	printf("\n\nUDP protocol\n\n");

	if( access(file_name, 0) == 0)
	{
		// send Exist
		strcpy(buf, "Exist");
		sendMessage(buf);
		Sleep(1000);
		resumeFile(f_name);
		return;
	}
	else
	{
		strcpy(buf, "None");
		sendMessage(buf);
		Sleep(1000);
	}

	printf("Receive this file : %s\n", file_name);
	
	char directory[1024] = "";
	int f_len = strlen(file_name), idx = 0;
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

	//receive file

	//receive N
	char number[10];
	strcpy(number, receiveMessage());
	int N = atoi(number);
	
	clock_t start_time = clock();
	int cnt = 1;
	double receive_size = 0.0;
	for(int i=0; i<N; i++)
	{
		strcpy(buf, receiveMessage());
		int size = strlen(buf);
		fwrite(buf, sizeof(buf[0]), size, outFile);
		receive_size += (double)size;
		if(cnt % 100 == 0) printf("%.2fMB/sec\n", (receive_size/(1024.0*1024.0)) / ((double)(clock() - start_time) / CLOCKS_PER_SEC));
		cnt++;
	}
	if(N != 0)
		printf("%.2fMB/sec\n", (receive_size/(1024.0*1024.0)) / ((double)(clock() - start_time) / CLOCKS_PER_SEC));
	else printf("0.0MB/sec\n");
	fclose(outFile);

	// receive server file hash value
	string server_file_hash_value = receiveMessage();

	if((outFile = fopen(file_name, "rb")) == NULL)
	{
		perror("outFile : ");
		exit(1);
	}

	// get receive file hash value
	string receive_file_hash_value = "";
	while( (len = fread(buf, 1, BUFSIZE, outFile)) )
	{
		buf[len] = 0;
		receive_file_hash_value = getHash(receive_file_hash_value + buf);
	}

	// receive server file size
	strcpy(buf, receiveMessage());
	int receive_file_size = atoi(buf);

	int my_file_size = getFileSize(file_name);


	//compare receive_hash_value, my_hash_value
	if(server_file_hash_value == receive_file_hash_value && receive_file_size == my_file_size)
		printf("The file was successfully received.\n");
	else
	{
		printf("%s file download failed... Please try again\n", file_name);
		corrupted_file_list.push_back(file_name);
		return;
	}
}

vector<string> UdpClientSocket::getCorruptedFileList()
{
	return corrupted_file_list;
}

void UdpClientSocket::clearCorruptedFileList()
{
	corrupted_file_list.clear();
}