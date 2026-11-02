"""Create an AUTH_PASSWORD_HASH value using scrypt (no password printed)."""
import getpass,hashlib,secrets
password=getpass.getpass("New workspace password: ")
salt=secrets.token_bytes(16);digest=hashlib.scrypt(password.encode(),salt=salt,n=2**14,r=8,p=1,dklen=32)
print("scrypt$"+salt.hex()+"$"+digest.hex())
