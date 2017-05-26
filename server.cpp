#include "UdpServerSocket.h"
#include "TcpServerSocket.h"

#define UDP_SERV_PORT 7718
#define TCP_SERV_PORT 7799

char data_list[][1024] = { {"1. train data\n"},
						{"2. test data\n"},
						{"3. README\n"} };

char data_path[][1024] = { {"data/train"},
						{"data/test"},
						{"data/README.txt"} };

int main()
{
	char* message;
	char buf[BUFSIZE];


	UdpServerSocket udp(UDP_SERV_PORT);
	TcpServerSocket tcp(TCP_SERV_PORT);

	udp.createSocket();
	udp.bindSocket();

	tcp.createSocket();
	tcp.bindSocket();
	tcp.listenSocket();

	Sleep(1000);
	tcp.acceptSocket();

	while(true)
	{
		printf("\nWaiting...\n");
		message = udp.receiveMessage();
		printf("%s\n", message);

		// send data list
		strcpy(buf, data_list[0]);
		for(int i=1; i<=2; i++)
			strcat(buf, data_list[i]);
		udp.sendMessage(buf);

		// receive select number
		strcpy(buf, udp.receiveMessage());
		int select = atoi(buf);

		// data path
		strcpy(buf, data_path[select-1]);

		udp.searchFiles(buf);

		// send file
		vector< pair<int, pair< string, string> > > file_list = udp.getFileList();
		int size = file_list.size();

		// send file list size 
		sprintf(buf, "%d", size);
		udp.sendMessage(buf);

		

		for(int i=0; i<size; i++)
		{
			int file_size = file_list[i].first;
			char* file_name = (char*)file_list[i].second.first.c_str();
			char* send_file_name = (char*)file_list[i].second.second.c_str();

			// send file size
			sprintf(buf, "%d", file_size);
			udp.sendMessage(buf);

			if(file_size <= 1024*64)
			{
				// send save file name 
				strcpy(buf, send_file_name);
				udp.sendMessage(buf);

				udp.sendFile(file_name);
			}
			else
			{
				// send save file name
				strcpy(buf, send_file_name);
				tcp.sendMessage(buf);

				tcp.sendFile(file_name);
			}
			
		}
	}

	return 0;

	/*
	// receive select number
	message = udp.receiveMessage();

	// send data list
	strcpy(buf, data_list[0]);
	for(int i=1; i<=2; i++)
		strcat(buf, data_list[i]);
	udp.sendMessage(buf);

	// receive select number
	strcpy(buf, udp.receiveMessage());
	int select = atoi(buf);

	// data path
	strcpy(buf, data_path[select-1]);




	// UDP
	if(!strcmp(message, "1"))
	{
		
		

		// send Files
		udp.searchFiles(buf);
		strcpy(buf, "CLEAR");
		udp.sendMessage(buf);
	}

	// TCP
	else if(!strcmp(message, "2"))
	{
		tcp.acceptSocket();
		strcpy(buf, data_list[0]);
		for(int i=1; i<=2; i++)
			strcat(buf, data_list[i]);
		tcp.sendMessage(buf);

		// receive select number
		strcpy(buf, tcp.receiveMessage());
		int select = atoi(buf);

		// data path
		strcpy(buf, data_path[select-1]);

		// send Files
		tcp.searchFiles(buf);
		
		strcpy(buf, "CLEAR");
		printf("buf : %s\n", buf);
		tcp.sendMessage(buf);
	}

	return 0;
	*/
}