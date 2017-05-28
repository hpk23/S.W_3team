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
#include <math.h>
#include "md5.h"
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 10240
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
	void clearFileList();
	string getHash(string md5Str); // get hash_value
	char* receiveMessage();
	void sendFile(char* file_name);
	void searchFiles(char* path);
	void resumeFile(string f_name);
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
	Sleep(100);
}

char* TcpServerSocket::receiveMessage()
{
	int mLen = recv(cliSock, buf, BUFSIZE, 0);
	buf[mLen] = 0;
	return buf;
}

void TcpServerSocket::clearFileList()
{
	file_list.clear();
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


void TcpServerSocket::resumeFile(string f_name)
{
	char* file_name = (char*)f_name.c_str();
	int len;

	FILE* file;

	// receive client exist file size
	strcpy(buf, receiveMessage());
	int client_file_size = atoi(buf);
	int my_file_size = getFileSize(file_name);

	if(client_file_size > my_file_size)
	{
		strcpy(buf, "strange file");
		sendMessage(buf);
		return;
	}
	else if(client_file_size == my_file_size)
	{
		strcpy(buf, "same");
		sendMessage(buf);
		return;
	}
	else
	{
		strcpy(buf, "OK");
		sendMessage(buf);
	}

	if( (file = fopen(file_name, "rb")) == NULL)
	{
		perror("file : ");
		exit(1);
	}

	//get server file hash value
	string server_file_hash_value = "";
	int amount = client_file_size;

	while((len = fread(buf, 1, min(BUFSIZE, amount), file)) && amount > 0)
	{
		buf[len] = 0;
		amount -= len;
		server_file_hash_value = getHash(server_file_hash_value + buf);
	}

	sendMessage((char*)server_file_hash_value.c_str());

	// receive file state
	strcpy(buf, receiveMessage());
	if(!strcmp(buf, "corrupt"))
	{
		printf("client file corrupted\n");
		return;
	}

	// send file

	Sleep(2000);
	int remain_size = my_file_size - client_file_size;
	int receive_size = ceil( (double)remain_size / BUFSIZE);
	sprintf(buf, "%d", receive_size);
	sendMessage(buf);
	server_file_hash_value = "";
	while((len = fread(buf, 1, BUFSIZE, file)))
	{
		buf[len] = 0;
		server_file_hash_value = getHash(server_file_hash_value + buf);
		sendMessage(buf);
		Sleep(20);
	}
	Sleep(500);

	// send server file hash value
	sendMessage((char*)server_file_hash_value.c_str());

	//send server file size
	sprintf(buf, "%d", my_file_size);
	sendMessage(buf);
	Sleep(500);

}

void TcpServerSocket::sendFile(char* file_name)
{
	printf("\nTCP protocol\n");

	strcpy(buf ,receiveMessage());
	printf("file exist : %s\n", buf);
	if(!strcmp(buf, "Exist"))
	{
		resumeFile(file_name);
		return;
	}

	printf("Transfer this file : %s\n", file_name);
	int len;
	FILE *file;

	if( (file = fopen(file_name, "rb")) == NULL)
	{
		perror("fopen : ");
		exit(1);
	}

	int my_file_size = getFileSize(file_name);

	// send file
	string server_file_hash_value = "";

	Sleep(2000);

	// send receive size
	int receive_size = ceil((double)my_file_size / BUFSIZE);
	sprintf(buf, "%d", receive_size);
	sendMessage(buf);

	int cnt = 1;
	while((len = fread(buf, 1, BUFSIZE, file)))
	{
		buf[len] = 0;
		server_file_hash_value = getHash(server_file_hash_value + buf);
		sendMessage(buf);
	}
	Sleep(500);

	// send server file hash value
	sendMessage((char*)server_file_hash_value.c_str());

	//send server file size
	sprintf(buf, "%d", my_file_size);
	sendMessage(buf);
	Sleep(500);

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
			memset(send_file_name, 0, sizeof(send_file_name));
		} 
	}
}