# This file contails the PostGresSQL code. Here we create the connecton to the PostgresSQL server online and 
# export our data to the data base. 


#%%
#!/usr/bin/python
import psycopg2
from configPostgresSQL import config

#%%
def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
       
	# close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

#%%
if __name__ == '__main__':
    connect()



# %%
## https://www.dataquest.io/blog/loading-data-into-postgres/
def CreateTable():
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        print('Crearing PostgreSQL database')
        cur.execute("""CREATE TABLE BostonEmpEarn(
                        name       text NOT NULL,
                        department text,
                        title      text,
                        regular   money,
                        retro     money,
                        other     money,
                        overtime  money,
                        injured   money,
                        detail    money,
                        quinn     money,
                        total     money,
                        zip        text,
                        year       text
                        
                        );
                        """)

        cur.execute('SELECT * FROM BostonEmpEarn')
        one = cur.fetchone()
        print(one)
       
	# close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.commit()
            conn.close()
            print('Database connection closed.')




#%%
if __name__ == '__main__':
    CreateTable()



    
# %%
## ImportTable: https://www.dataquest.io/blog/loading-data-into-postgres/
def ImportTable():
    import csv
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        print('Importing PostgreSQL database - Pusing to cloud (elephantSQL)')
        
        
        with open('data/EmpEarn.csv', 'r',encoding="utf-8") as f:
            # reader = csv.reader(f)
            # next(reader) # Skip the header row.
            # for row in reader:
            #     cur.execute(
            #     "INSERT INTO BostonEmpEarn VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            #     row
            # )
            # Notice that we don't need the `csv` module.
            next(f) # Skip the header row.
            cur.copy_from(f, 'BostonEmpEarn', sep=',')

        conn.commit()

        cur.execute('SELECT * FROM BostonEmpEarn')
        one = cur.fetchone()
        print(one)
       
	# close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        if conn is not None:
            conn.rollback()
    finally:
        if conn is not None:
            
            conn.close()
            print('Database connection closed.')



# %%
if __name__ == '__main__':
    ImportTable()

# %%
