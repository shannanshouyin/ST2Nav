"""Create new local service credentials; never contains a shipped secret."""
import argparse, os, re, secrets, subprocess, tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hostname',required=True,help='DNS name used by the robot, matching the certificate')
    p.add_argument('--state-dir',type=Path,default=Path.home()/'.local/share/etpr1')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*',a.hostname):p.error('Use a DNS hostname')
    a.state_dir.mkdir(parents=True,exist_ok=True)
    paths=[a.state_dir/n for n in ['server.key','server.crt','token']]
    if any(x.exists() for x in paths):raise SystemExit('Credentials already exist; refusing to overwrite')
    old=os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix='navigation-auth-') as work:
            config=Path(work)/'openssl.cnf'
            config.write_text('[req]\ndistinguished_name=dn\nx509_extensions=ext\n[dn]\nCN='+a.hostname+
                              '\n[ext]\nsubjectAltName=DNS:'+a.hostname+
                              '\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n',encoding='ascii')
            result=subprocess.run(['openssl','req','-x509','-newkey','rsa:3072','-nodes','-days','365',
                                   '-keyout',str(paths[0]),'-out',str(paths[1]),'-subj','/CN='+a.hostname,
                                   '-config',str(config)],capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError('OpenSSL failed: '+result.stderr)
        paths[2].write_text(secrets.token_urlsafe(48)+'\n',encoding='ascii')
    except Exception:
        for path in paths:path.unlink(missing_ok=True)
        raise
    finally:os.umask(old)
    print('Credentials created. Transfer only server.crt and token to the robot over a trusted channel.')
if __name__=='__main__':main()
