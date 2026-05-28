# utils/download_real_data.py
import os
import urllib.request
import gzip
import shutil

class RealDataDownloader:
    """Download real-time traffic data"""
    
    # MAWI traffic data (PCAP format)
    MAWI_BASE_URL = "https://mawi.wide.ad.jp/mawi/samplepoint-F/2022/"
    MAWI_SAMPLES = [
        "202209011400.pcap.gz",  # 2022.9
    ]
    
    # SNDlib Traffic Matrix
    SNDLIB_BASE_URL = "http://sndlib.zib.de/download/"
    SNDLIB_NETWORKS = [
        "abilene",
        "geant",
        "germany50",
        "nobel-germany",
    ]
    
    def __init__(self, data_dir='data/real_traces'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'mawi'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'sndlib'), exist_ok=True)
    
    def download_mawi_sample(self, sample_name=None):
        """Download MAWI traffic samples"""
        if sample_name is None:
            sample_name = self.MAWI_SAMPLES[0]
        
        url = f"{self.MAWI_BASE_URL}{sample_name}"
        output_path = os.path.join(self.data_dir, 'mawi', sample_name)
        
        print(f"Downloading MAWI sample from {url}...")
        print("Note: MAWI files are large (several GB). This may take a while.")
        
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded to {output_path}")
            
            # 解压
            if output_path.endswith('.gz'):
                print("Extracting...")
                with gzip.open(output_path, 'rb') as f_in:
                    with open(output_path[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Extracted to {output_path[:-3]}")
            
            return output_path
        except Exception as e:
            print(f"Download failed: {e}")
            print("You can manually download from: https://mawi.wide.ad.jp/mawi/")
            return None
    
    def download_sndlib_network(self, network_name):
        """Download SNDlib network topology and traffic matrix"""
        # SNDlib provides topology and traffic matrices in XML format
        url = f"{self.SNDLIB_BASE_URL}directed-{network_name}.tar.gz"
        output_path = os.path.join(self.data_dir, 'sndlib', f'{network_name}.tar.gz')
        
        print(f"Downloading SNDlib network {network_name}...")
        
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded to {output_path}")
            
            # Unzip
            import tarfile
            with tarfile.open(output_path, 'r:gz') as tar:
                tar.extractall(os.path.join(self.data_dir, 'sndlib'))
            print(f"Extracted to {self.data_dir}/sndlib/{network_name}")
            
            return output_path
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"You can manually download from: http://sndlib.zib.de/home.action")
            return None
    
    def download_all(self):
        """Download all data"""
        print("=" * 60)
        print("Downloading Real Traffic Data")
        print("=" * 60)
        
        # Download SNDlib networks
        for network in self.SNDLIB_NETWORKS:
            self.download_sndlib_network(network)
        
        print("\nNote: MAWI pcap files are very large.")
        print("Please download manually from: https://mawi.wide.ad.jp/mawi/")
        print("Recommended: samplepoint-F/2022/202209011400.pcap.gz")


if __name__ == '__main__':
    downloader = RealDataDownloader()
    downloader.download_all()