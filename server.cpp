//#include "UdpServerSocket.h"
#include "TcpServerSocket.h"
#define PORT 7080

int main()
{
	char* message;
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

	strcpy(buf, "D:/S.W_3team/step3.txt");
	tcp.sendFile(buf);

	return 0;
	/*char* message;
	char buf[BUFSIZE];

	UdpServerSocket udp(PORT);
	udp.createSocket();
	udp.bindSocket();

	message = udp.receiveMessage();
	printf("%s\n", message);

	strcpy(buf, "client/step3.txt"); // 파일이름 복사
	udp.sendMessage(buf); // 파일이름 클라이언트에게 보내기

	strcpy(buf, "D:/S.W_3team/step3.txt"); //보낼 파일 경로 buf에 복사
	udp.sendFile(buf); // 파일 보내기

	return 0;*/
}