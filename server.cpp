#include "UdpServerSocket.h"
#include "TcpServerSocket.h"
#define UDP_PORT 7709
#define TCP_PORT 7710

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

	UdpServerSocket udp(UDP_PORT);
	TcpServerSocket tcp(TCP_PORT);

	udp.createSocket();
	udp.bindSocket();

	tcp.createSocket();
	tcp.bindSocket();
	tcp.listenSocket();

	message = udp.receiveMessage();

	// UDP
	if(!strcmp(message, "1"))
	{
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
		printf("buf : %s\n", buf);

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


	/*char* message;
	char buf[BUFSIZE];

	TcpServerSocket tcp(PORT);
	tcp.createSocket();
	tcp.bindSocket();
	tcp.listenSocket();
	tcp.acceptSocket();

	message = tcp.receiveMessage();
	printf("%s\n", message);

	strcpy(buf, "client/step3.txt");
	tcp.sendMessage(buf);

	strcpy(buf, "files/step3.txt");
	tcp.sendFile(buf);
	strcpy(buf, "./");

	return 0;*/
	/*char* message;
	char buf[BUFSIZE];

	UdpServerSocket udp(PORT);
	udp.createSocket();
	udp.bindSocket();

	message = udp.receiveMessage();
	printf("%s\n", message);

	// send file_name to the client
	strcpy(buf, "client/step3.txt"); 
	udp.sendMessage(buf); 

	// send file to the client
	strcpy(buf, "files/step3.txt"); //보낼 파일 경로 buf에 복사
	udp.sendFile(buf); // 파일 보내기

	return 0;*/
}