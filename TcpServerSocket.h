#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <winsock2.h>
#include <fstream>
#include <windows.h>
#include <string>
#include <iostream>
#include <dirent.h>
#include <vector>
#include <utility>
#include "md5.h"
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024 * 10
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

	vector< pair<int, pair< string, string> > > file_list;
public :
	TcpServerSocket(int port);
	void createSocket();
	void bindSocket();
	void listenSocket();
	void acceptSocket();
	void closeSocket();
	void sendMessage(char* message);
	string getHash(string md5Str); // get hash_value
	char* receiveMessage();
	void sendFile(char* file_name);
	void searchFiles(char* path);
	int getFileSize(char* file_name);
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

void TcpServerSocket::closeSocket()
{
	closesocket(servSock);
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

int TcpServerSocket::getFileSize(char* file_name)
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
	printf("\nTCP protocol\n");
	printf("Transfer this file : %s\n", file_name);
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

void TcpServerSocket::searchFiles(char* path)
{
	DIR *dp;
	struct dirent *dent;
	char temp[1024];
	char send_file_name[BUFSIZE] = "";

	if((dp = opendir(path)) == NULL)
	{ 
		char temp_path[1024];
		strcpy(temp_path, path);
		char* ptr = strtok(temp_path, "/");
		while(ptr != NULL)
		{
			if(!strcmp(ptr, "data"))
			{
				ptr = strtok(NULL, "/");
				break;
			}
			ptr = strtok(NULL, "/");
		}

		while(ptr != NULL)
		{
			strcat(send_file_name, ptr);
			ptr = strtok(NULL, "/");
		}

		int file_size = getFileSize(path);
		file_list.push_back(make_pair(file_size, make_pair(path, send_file_name)));
		//sendMessage(send_file_name);
		//sendFile(path);
	}

	while((dent = readdir(dp)))
	{
		if(dent->d_name[0] == '.') continue;

		sprintf(temp, "%s/%s", path, dent->d_name);
		if(opendir(temp) != NULL) searchFiles(temp);

		else
		{
			char temp_path[1024];
			strcpy(temp_path, path);
			char* ptr = strtok(temp_path, "/");
			while(ptr != NULL)
			{
				if(!strcmp(ptr, "data"))
				{
					ptr = strtok(NULL, "/");
					break;
				}
				ptr = strtok(NULL, "/");
			}

			while(ptr != NULL)
			{
				strcat(send_file_name, ptr);
				ptr = strtok(NULL, "/");
			}
			strcat(send_file_name, "/");
			strcat(send_file_name, dent->d_name);
			
			int file_size = getFileSize(temp);
			file_list.push_back(make_pair(file_size, make_pair(temp, send_file_name)));
			//sendMessage(send_file_name);
			memset(send_file_name, 0, sizeof(send_file_name));
			//sendFile(temp);
		} 
	}
}