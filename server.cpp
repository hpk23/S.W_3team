#include "UdpServerSocket.h"

#define PORT 7080

int main()
{
	UdpServerSocket udp(PORT);
	udp.createSocket();
	udp.bindSocket();
	while(1)
	{
		char* message;
		char buf[BUFSIZE];

		printf("Waiting...\n");

		message = udp.receiveMessage();
		printf("%s\n", message);

		strcpy(buf, "client/step3.txt"); // 파일이름 복사
		udp.sendMessage(buf); // 파일이름 클라이언트에게 보내기

		strcpy(buf, "D:/S.W_3team/step3.txt"); //보낼 파일 경로 buf에 복사
		udp.sendFile(buf); // 파일 보내기
	}

	return 0;
}