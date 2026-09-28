#ifndef SLORADO_SERVER_SOCKET_H
#define SLORADO_SERVER_SOCKET_H

#include <stdint.h>

/*
 * TCP is a byte stream, so this module gives the application a message
 * abstraction. Every message is a 4-byte network-order (big-endian) payload
 * length followed by exactly that many payload bytes. This matches readfish's
 * readfish/plugins/_network_io.py, which is the client this server talks to.
 */

/******************************* Blocking Send and Receive Functions***********************************/

/*  Send one framed message.

    arguments : socket - connected socket
                buffer - payload bytes
                length - payload size in bytes

    The length prefix and payload are pushed with a single writev() so the
    response is not split into two small segments. Blocks until everything is
    sent. Returns 0 on success and -1 on error (errno is set).
*/
int send_full_msg(int socket, const void *buffer, uint32_t length);

/*  Receive one framed message into a freshly allocated buffer.

    arguments : socket - connected socket
                buffer - on success, points to a malloc'd buffer of *length bytes (caller frees)
                length - on success, the payload size in bytes
                max_length - reject messages whose advertised size exceeds this

    Blocks until the whole message has arrived. Returns
         0 on success
         1 on an orderly peer disconnect before any byte of the message
        -1 on an I/O or allocation error (errno is set)
        -2 if the advertised size exceeds max_length
*/
int recv_full_msg(int socket, void **buffer, uint32_t *length, uint32_t max_length);


/***************************Server side*******************************************************/

/*Create a TCP socket on PORT
  Internally it performs binding and listening system calls
  Returns the listening socket
  */
int TCP_server_init(int PORT);

/*Accept a client through the listening socket "listenfd"
  Return the connection socket, or -1 on error (errno is set).
*/
int TCP_server_accept_client(int listenfd);

/*Disconnect a connected client*/
void TCP_server_disconnect_client(int connectfd);

/*Close down the listening socket*/
void TCP_server_shutdown(int listenfd);



/********************************Client side***************************************************/

/* Connect to a TCP server at the port "PORT" at IP address "ip"
   returns the connection socket
*/
int TCP_client_connect(const char *ip, int PORT);

/* Disconnect*/
void TCP_client_disconnect(int socketfd);

#endif
