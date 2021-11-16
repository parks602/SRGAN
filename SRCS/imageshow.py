import numpy as np
import matplotlib.pyplot as plt
from utils import ReadPPM, ReadASCII
from datetime import datetime, timedelta

def main():
    date  = '2021040200'
    ftime = '9'
    var   = 'T3H'
    fmt   = "%Y%m%d%H"
    srgan_name = '/data/home/rndkoa/2021/SRGAN/DAOU/DFS_SHRT_STD_GDPS_SRGN_%s.%s'%(var, date)

    ldapsdate = datetime.strptime(date, fmt) + timedelta(hours = int(ftime))
    ldapsdate = datetime.strftime(ldapsdate, fmt)

    ldaps_name  =  '/data/home/rndkoa/2021/LDAPS/DAOU/DFS_SHRT_STD_ANL_1KM_%s.%s'%(var, ldapsdate)
    fig  = plt.figure()
    rows = 1
    cols = 2
    print(srgan_name)
    print(ldaps_name)
    print(srgan_name)
    ppm  = ReadPPM(srgan_name)
    ppm  = ppm.GetData()
    ppm  = ppm[ftime.zfill(3)]['data']
    ppm  = np.array(ppm)
    
    ppm  = np.reshape(ppm , (745, 1265))
    print(ppm.shape)

    ldaps = ReadASCII(ldaps_name)
    ldaps = ldaps.getData()
    import math
    print(math.sqrt(np.mean(np.square(ldaps-ppm))))
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(ppm, vmin=-30, vmax=30)
    ax1.set_title('SRGAN')
    ax1.axis('off')

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(ldaps,  vmin=-30, vmax=30)
    ax2.set_title('LDAPS')
    ax2.axis('off')

    plt.show()
if __name__ == '__main__':
    main()
