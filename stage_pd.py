from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from cv2 import *
import Person_det_track
import detector
import keras
import os, sys, time, socket
import threading
import argparse
import numpy as np
import cv2

modelsdir = 'models'
default_server_port = 9252

"""
Predict class of an image
"""

# camera: parameter to specify the camera is used. If False, a single image is passed to predict people.
def detectPeople(imagefile,camera=False):

    if isinstance(imagefile, str):
        inp = inputImage(imagefile)
    else:
        inp = imagefile

    if inp is not None:
        # It returns:
        #    - bounding boxes
        #    - confidence
        det = detector.PersonDetector()

        # We need the confidence.
        # If the image comes from the client, it is already a numpy array, otherwise we must transform it
        img_full_np = inp
        if isinstance(imagefile, str):
            img_full_np = np.asarray(inp)

        (_,confidence) = det.get_localization(img_full_np)

        img, howMany = Person_det_track.pipeline(inp,det,camera)
        return (img, howMany, confidence)

    else:
        return (0, 'error')


"""
Load an image and return input data for the network
"""


def inputImage(imagefile):
    try:
        #IMREAD_GRAYSCALE
        gray = cv2.imread(imagefile, IMREAD_COLOR)
        gray = np.array(gray, dtype='uint8')
        return gray
    except:
        return None


"""
Load a trained model
"""


def loadModel(modelname):
    global modelsdir
    filename = os.path.join(modelsdir, '%s.h5' % modelname)
    try:
        model = keras.models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" % filename)
    except OSError:
        print("\nModel file %s not found!!!\n" % modelname)
        model = None
    return model


class ModelServer(threading.Thread):

    def __init__(self, port):
        threading.Thread.__init__(self)

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(3)  # timeout when listening (exit with CTRL+C)

        # Bind the socket to the port
        server_address = ('', port)
        self.sock.bind(server_address)
        self.sock.listen(1)

        print("Server running on port %d" % port)

        self.dorun = True  # server running
        self.connection = None  # connection object

    def stop(self):
        self.dorun = False

    def connect(self):
        connected = False
        while (self.dorun and not connected):
            try:
                self.connection, client_address = self.sock.accept()
                self.connection.settimeout(3)
                connected = True
                print('Connection from %s' % str(client_address))
            except:
                pass

                # buf may contain a first chunk of data

    def recvall(self, count, chunk):
        buf = chunk
        count -= len(buf)
        while count > 0:
            newbuf = self.connection.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def run(self):

        imgsize = -1
        res = 'none 0.0'
        while (self.dorun):
            self.connect()  # wait for connection
            try:
                # Receive data
                while (self.dorun):
                    try:
                        data = self.connection.recv(256)
                        data = data.strip()
                    except socket.timeout:
                        data = "***"
                    except Exception as e:
                        print(e)
                        data = None

                    buf = b''
                    if (type(data) != str):
                        k = data.find(b'\n')
                        if (k < 0):
                            data = data.decode('utf-8')
                        elif (len(data) > k + 1):
                            buf = data[k + 2:]
                            data = data[0:k].decode('utf-8')

                    if (data != None and data != "" and data != "***"):
                        self.received = data
                        print('Received: %s' % data)
                        v = data.split(' ')
                        if v[0]=='GETRESULT':
                            ressend = (res + '\n\r').encode('UTF-8')
                            self.connection.send(ressend)

                        elif v[0] == 'EVAL' and len(v) > 1:


                            (people_detected, howMany, confidence) = detectPeople(v[1])

                            # Without resizing the window, it will fit the whole screen
                            cv2.namedWindow("detected", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("detected", 959, 1280)

                            cv2.imshow("detected", people_detected)
                            cv2.waitKey(6000)
                            cv2.destroyAllWindows()

                            res = "People detected!"
                            ressend = (res + '\n\r').encode('UTF-8')
                            self.connection.send(ressend)
                        elif v[0]=='RGB' and len(v) >= 3:
                            print("\n---------Predicting faces----------\n")

                            img_width = int(v[1])
                            img_height = int(v[2])
                            img_size = img_height * img_width * 3

                            print("RGB image size: %d" %img_size)
                            buf = self.recvall(img_size, buf)

                            if buf is not None:
                                print("Image received with size: %d" %len(buf))
                                img_rcv = np.fromstring(buf, dtype='uint8')
                                img_rcv = img_rcv.reshape((img_height, img_width, 3))

                                # Image as array
                                inp = np.asarray(img_rcv)

                                # Prediction
                                (people_detected,howMany,confidence) = detectPeople(inp)

                                # Without resizing the window, it will fit the whole screen
                                # cv2.namedWindow("detected", cv2.WINDOW_NORMAL)
                                # cv2.resizeWindow("detected", img_width, img_height)
                                #
                                # cv2.imshow("detected", people_detected)
                                # cv2.waitKey(6000)
                                # cv2.destroyAllWindows()

                                # The format of the response will be:
                                #  [howMany confidence[0] confidence[1] .. confidence[N]]
                                if (howMany > 0):
                                    res = ''.join( (str(howMany),' '))
                                    for i in range(howMany):
                                        res += str(confidence[i]) + ' '
                                    res.rstrip()

                                ressend = (res + '\n\r').encode('UTF-8')
                                self.connection.send(ressend)
                        else:
                            print('Received: %s' % data)
                    elif (data == None or data == ""):
                        break
            finally:
                print('Connection closed.')
                # Clean up the connection
                if (self.connection != None):
                    self.connection.close()
                    self.connection = None

    # wait for Keyboard interrupt
    def spin(self):
        while (self.dorun):
            try:
                time.sleep(120)
            except KeyboardInterrupt:
                print("Exit")
                self.dorun = False


"""
Start prediction server
"""


def startServer(port):
    print("Starting server on port %d" % port)
    mserver = ModelServer(port)
    mserver.start()
    mserver.spin()
    mserver.stop()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default=False, action='store_true',
                        help='Start in server mode')
    parser.add_argument('-server_port', type=int, default=default_server_port,
                        help='server port (default: %d)' % default_server_port)
    parser.add_argument("-predict", type=str, default=None,
                        help="Image file to predict")

    args = parser.parse_args()


    if (args.server):
        print("\n -----Person detection:------\n")
        startServer(args.server_port)
    elif (args.predict != None):
        print("\n -----Person detection:------\n")
        person_detection = detectPeople(args.predict)

        # Without resizing the window, it will fit the whole screen
        cv2.namedWindow("detected",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("detected",(959,1280))
        cv2.imshow("detected",person_detection)
        cv2.waitKey(0)
        cv2.imwrite("prova.jpg",person_detection)
    else:
        print("Please specify a model name and an operation to perform.")
        sys.exit(1)