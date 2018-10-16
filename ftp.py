import ftplib
import socket

if __name__ == "__main__":
    while(True):
        
        try:
            ftp = ftplib.FTP('140.118.155.111', timeout=0.5)
            ftp.login('ccnlab', 'balncc')
            ftp.cwd('/')
            print('get')
            ftp.quit()
            break
        except ftplib.error_perm as e:
            print('Error {}'.format(e.args[0][:3]))
        except ConnectionRefusedError:
            print('not available')
        except socket.gaierror as e:
            print("Address-related error connecting to server: %s" % e)

            