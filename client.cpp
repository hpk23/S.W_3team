#include "UdpClientSocket.h"
#include "TcpClientSocket.h"

#define UDP_CLIENT_PORT 6500
#define TCP_CLIENT_PORT 6600

#define UDP_SERV_PORT 7718
#define TCP_SERV_PORT 7799

int main()
{
	char* message;
	char IP[100], buf[BUFSIZE], file_name[100];
	int select;

	strcpy(IP, "127.0.0.1");

	UdpClientSocket udp(UDP_CLIENT_PORT, IP, UDP_SERV_PORT);
	TcpClientSocket tcp(TCP_CLIENT_PORT, IP, TCP_SERV_PORT);

	udp.createSocket();

	
	tcp.createSocket();
	tcp.connectSocket();
	Sleep(1000);

	strcpy(buf, "I want to download a file");
	udp.sendMessage(buf);

	// receive data list
	message = udp.receiveMessage();

	// select data
	while(true)
	{
		printf("%s\n", message);
		printf("Please select a number : ");
		scanf("%d", &select);
		if(select == 1 || select == 2 || select == 3) break;
		else printf("Invalid number\n\n");
	}

	// send select number
	sprintf(buf, "%d", select);
	udp.sendMessage(buf);

	//receive file list size
	strcpy(buf, udp.receiveMessage());
	int size = atoi(buf);

	//receive file
	for(int i=0; i<size; i++)
	{
		// receive file size
		strcpy(buf, udp.receiveMessage());
		int file_size = atoi(buf);

		if(file_size <= 1024 * 64)
		{
			char* file_name = udp.receiveMessage();
			udp.receiveFile(file_name);
		}
		else
		{
			char* file_name = tcp.receiveMessage();
			tcp.receiveFile(file_name);
		}
	}

	vector<string> fail_file_list = tcp.getCorruptedFileList();
	size = fail_file_list.size();
	printf("\n----------download fail file----------\n");
	int cnt = 1;
	for(int i=0; i<size; i++)
		printf("%d. %s\n", cnt++, (char*)fail_file_list[i].c_str());
	fail_file_list = udp.getCorruptedFileList();
	size = fail_file_list.size();
	for(int i=0; i<size; i++)
		printf("%d. %s\n", cnt++, (char*)fail_file_list[i].c_str());

	return 0;
}