#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include "socket.h"


/*****************Internal Function prototypes that are not included in socket.h ***********************/

/*Die on error. Print the error and exit if the return value of the previous function is -1*/
static void errorCheck(int ret, const char *msg);

/*Disable Nagle so small responses are not held back waiting for an ACK*/
static void set_nodelay(int socket);

/*Receive length number of bytes to the buffer. Returns 0 on success, 1 on EOF, -1 on error*/
static int recv_all(int socket, void *buffer, uint64_t length);


/*******************************Blocking send and receive***********************************/

/*Send the 4-byte network-order length and then the payload, in one sendmsg() where possible.
  MSG_NOSIGNAL turns a vanished peer into EPIPE instead of a process-killing SIGPIPE.*/
int send_full_msg(int socket, const void *buffer, uint32_t length)
{
    uint32_t prefix = htonl(length);
    struct iovec iov[2];
    iov[0].iov_base = &prefix;
    iov[0].iov_len = sizeof(prefix);
    iov[1].iov_base = (void *)buffer;
    iov[1].iov_len = length;

    struct iovec *cur = iov;
    int n_iov = length ? 2 : 1;
    while (n_iov > 0) {
        struct msghdr msg;
        memset(&msg, 0, sizeof(msg));
        msg.msg_iov = cur;
        msg.msg_iovlen = n_iov;
        ssize_t sent = sendmsg(socket, &msg, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        //advance over whatever was sent, which may end part way through an iovec
        while (n_iov > 0 && (size_t)sent >= cur->iov_len) {
            sent -= cur->iov_len;
            cur++;
            n_iov--;
        }
        if (n_iov > 0) {
            cur->iov_base = (char *)cur->iov_base + sent;
            cur->iov_len -= sent;
        }
    }
    return 0;
}

/*First receive the number of bytes to be received. Then receive the actual data to a new buffer. */
int recv_full_msg(int socket, void **buffer, uint32_t *length, uint32_t max_length)
{
    *buffer = NULL;
    *length = 0;

    uint32_t prefix = 0;
    int ret = recv_all(socket, &prefix, sizeof(prefix));
    if (ret != 0) return ret;

    uint32_t expected_length = ntohl(prefix);
    if (expected_length > max_length) {
        fprintf(stderr, "Peer advertised a %u byte message (limit %u)\n", expected_length, max_length);
        return -2;
    }

    void *data = malloc(expected_length ? expected_length : 1);
    if (data == NULL) return -1;

    ret = recv_all(socket, data, expected_length);
    if (ret != 0) {
        free(data);
        return -1; //EOF part way through a message is an error, not an orderly close
    }

    *buffer = data;
    *length = expected_length;
    return 0;
}

/***************************Server side*******************************************************/


/*Create a TCP socket, bind and then listen*/
int TCP_server_init(int PORT){

	//initialize variables
	int listenfd,ret;
	struct sockaddr_in server;
	memset(&server, 0, sizeof(server));

	server.sin_family = AF_INET;
	server.sin_addr.s_addr = htonl(INADDR_ANY);
	server.sin_port = htons(PORT);

	//open socket bind and listen
	listenfd = socket(AF_INET,SOCK_STREAM,0);
	errorCheck(listenfd,"Cannot create socket");

    /****sophisticated way of binding*/

    //This lets a port be used as soon as the using program ends. Otherwise you will get "Cannot bind : Address is used"
    //for few minutes after the program that previously used the port exists
    int enable = 1;
    if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0){
        perror("setsockopt(SO_REUSEADDR) failed");
    }

    //try binding
    ret=-1;
    while(ret==-1){
        ret=bind(listenfd,(struct sockaddr *)&server, sizeof(server));
        if (ret==-1){
            if (errno != EADDRINUSE) break;
            fprintf(stderr,"Cannot bind : Address is used. Trying agin..\n");
            sleep(1);
        }
    }

    /*** end of the sophisticated way*/

    errorCheck(ret,"Cannot bind");
    fprintf(stderr,"Binding successfull... port : %d\n",PORT);


    //now listen
    ret=listen(listenfd,5);
	errorCheck(ret,"Cannot listen");
	fprintf(stderr,"Listening to port %d...\n",PORT);

    return listenfd;

}

/*Accept a client*/
int TCP_server_accept_client(int listenfd){

    int connectfd;

    struct sockaddr_in client;
	socklen_t clientlen = sizeof(client);

	//accept connections
	connectfd = accept(listenfd,(struct sockaddr *)&client, &clientlen);
	if (connectfd < 0) return -1;

    set_nodelay(connectfd);

    //get ip to string
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client.sin_addr, ip, sizeof(ip));
    fprintf(stderr,"Client %s:%d connected\n",ip,ntohs(client.sin_port));

    return connectfd;

}

/*Disconnect a connected client*/
void TCP_server_disconnect_client(int connectfd){
	//close sockets
	int ret=close(connectfd);
	if (ret == -1) perror("Cannot close socket");
	fprintf(stderr,"Client disconnected\n");
}

/*Close down the listening socket*/
void TCP_server_shutdown(int listenfd){
    //we do not need the listening socket now
    int ret=close(listenfd);
    errorCheck(ret,"Cannot close socket");
}


/********************************Client side***************************************************/


/* Connect to a TCP server at PORT at ip*/
int TCP_client_connect(const char *ip, int PORT){

	//initializing the server address and assigning port numbers
	struct sockaddr_in server;
	memset(&server, 0, sizeof(server));
	server.sin_family = AF_INET;
	server.sin_port = htons(PORT);
	if (inet_pton(AF_INET, ip, &server.sin_addr) != 1) {
		fprintf(stderr, "Invalid IPv4 address: %s\n", ip);
		exit(EXIT_FAILURE);
	}

    //try connecting until successful. this is better than trying once and giving up
    //a failed connect() leaves the socket unusable, so each attempt needs a fresh one
    int socketfd = -1;
 	int ret=-1;
    while(ret==-1){
        socketfd = socket(AF_INET,SOCK_STREAM,0);
        errorCheck(socketfd,"Cannot create socket");
        ret=connect(socketfd,(struct sockaddr *)&server, sizeof(server));
        if (ret==-1) {
            close(socketfd);
            sleep(1);
        }
    }

    set_nodelay(socketfd);
    fprintf(stderr,"Connected to server \n");

    return socketfd;

}


/* Disconnect*/
void TCP_client_disconnect(int socketfd){

	//closing sockets
	int ret = shutdown(socketfd,SHUT_RDWR);
    if (ret == -1) perror("Cannot disconnect");
 	ret = close(socketfd);
    errorCheck(ret,"Cannot close socket");
}




/************************************Internal Functions*****************************************/

/*Die on error. Print the error and exit if the return value of the previous function is -1*/
static void errorCheck(int ret, const char *msg){
	if(ret==-1){
		perror(msg);
		exit(EXIT_FAILURE);
	}
}

static void set_nodelay(int socket){
    int enable = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable)) < 0){
        perror("setsockopt(TCP_NODELAY) failed");
    }
}

/*Receive length number of bytes to the buffer. This is a blocking call. Make sure that buffer is big enough.*/
static int recv_all(int socket, void *buffer, uint64_t length)
{
    char *ptr = (char*) buffer;
    while (length > 0)
    {
        ssize_t i = recv(socket, ptr, length, 0);
        if (i == 0) return 1;
        if (i < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        ptr += i;
        length -= i;
    }
    return 0;
}
