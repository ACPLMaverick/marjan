#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <string.h>

using namespace std;

//funkcja koduje zawartosc pliku
void kodowanie(int h[][16], FILE*do_kodowania, FILE*zakodowany)
{
     int znak;
     int wiad[8];
     int kontrol[8];



     while((znak=fgetc(do_kodowania))!=EOF)
     {
          for(int i=0; i<8; i++)
             kontrol[i]=0;
          //do tablicy wiadomosci zapisujemy binarny odpowiednik wczytanego znaku
          for(int i=7; i>-1; i--)
          {
                  wiad[i]=znak%2;
                  znak=znak/2;
          }

          //uzupe³niamy tablicê bitów kontroli parzystoœci
          for(int i=0; i<8; i++)
          {
                  for(int j=0; j<8; j++)
                          kontrol[i]+=wiad[j]*h[i][j];

                  kontrol[i]%=2;
          }

          //zapisujemy zawartosc obu tablic do pliku zakodowany
          for(int i=0; i<8; i++)
                  fputc(wiad[i]+48, zakodowany);
          for(int i=0; i<8; i++)
                  fputc(kontrol[i]+48, zakodowany);
          fputc('\n', zakodowany);
     }
     cout<<"Zawartosc pliku zostala pomyslnie zakodowana\n";
}


//funkcja sprawdza poprawnosc przeslanych danych
void sprawdzanie(int h[][16], FILE*zakodowany, FILE*odkodowany)
{
     int znak;
     int tab[16];             //tablica, do ktorej zapisywany jest wiersz pliku zakodowany
     int blad[8];             //tablica bledow danego znaku
     int licznik=0;
     int licznik_znakow=1;
     int ilosc_bledow=0;

     while((znak=fgetc(zakodowany))!=EOF)
     {
          if(znak!='\n')
          {

               tab[licznik]=znak-48;  //nalezy odjac 48 bo to tablica int a nie char
               licznik++;

          }
          else
          {
              licznik=0;
              int numer_bitu1;       //nr bitu z bledem
              int numer_bitu2;       //nr drugiego bitu z bledem w przypadku wystapienia dwoch bledow
              for(int i=0; i<8; i++)
              {
                      blad[i]=0;
                      for(int j=0; j<16; j++)
                              blad[i]+=tab[j]*h[i][j];
                      blad[i]%=2;

                      if(blad[i]==1)
                        ilosc_bledow=1;
              }
              if(ilosc_bledow!=0)
              {
              cout<<"Nr bitow liczone sa od 0\n";
              int jest = 0;
                 //tu trzeba sprawdzic, czy jest tylko jeden blad
                 for(int i=0;i<15;i++)
                   for(int j=i+1;j<16;j++)
                   {
                     jest = 1;
                     for(int k=0;k<8;k++)
                       if (blad[k] != h[k][i] ^ h[k][j])
                       {
                         jest = 0;
                         break;
                       }
                     if (jest == 1)
                     {
                         numer_bitu1 = i;
                         numer_bitu2 = j;
                         cout << "Wystapily dwa bledy w "<<licznik_znakow<<" znaku na " << i << " i " <<j<<" bicie\n";
                         tab[numer_bitu1]=!tab[numer_bitu1];
                         tab[numer_bitu2]=!tab[numer_bitu2];
                         ilosc_bledow=2;
                         i = 16;
                         break;
                     }
                   }
                         if(ilosc_bledow==1)
                         {
                                  for(int i=0; i<16; i++)
                                  {
                                          for(int j=0; j<8; j++)
                                          {
                                                  if(h[j][i]!=blad[j])
                                                  break;              //jesli choc jeden element kolumny tab h rozni sie
                                                                      //od elementu tab blad to przejdz do kolejnej kolumny
                                                  if(j==7)
                                                  {
                                                          numer_bitu1=i;
                                                          cout<<"Blad wystapil w "<<licznik_znakow<<" znaku na "<<i<<" bicie"<<endl;
                                                          tab[i]=!tab[i];
                                                          i=16;

                                                  }
                                          }
                                  }
                                  ilosc_bledow=0;
                         }
                         if(ilosc_bledow==2)
                         {

                         }
              }

              licznik=0;
              licznik_znakow++;
              ilosc_bledow=0;

              //zapisujemy do pliku
              int a=128;
              char kod=0;
              for(int i=0; i<8; i++)
              {
                      kod+=a*tab[i];
                      a/=2;
              }
              fputc(kod,odkodowany);

          }

     }


}



//glowny kod programu
int main()
{
    string pom;               //pomocnicza do utworzenia nazwy pliku do zakodowania z rozszerzeniem
    string nazwa;             //nazwa pliku do zakodowania
    int wybor;               //decyzja o tym co program ma robic

   	int h[8][16] = {{1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0},
                    {1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0},
                    {1,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0},
                    {0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0},
                    {1,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0},
                    {1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0},
                    {0,1,1,1,1,0,1,1,0,0,0,0,0,0,1,0},
                    {1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,1}};


int n=1;
while (n!=0){
 cout<<endl<<"Wybierz jedna z opcji:"<<endl;
    cout<<"1 - kodowanie wybranego pliku"<<endl;
    cout<<"2 - odkodowanie"<<endl;
    cout<<"3 - AUTORZY"<<endl;
    cout<<"4 - KONIEC"<<endl;
    cin>>wybor;
//switch(wybor)
//{

//case 1:
         if (wybor==1)
         {
              FILE* do_kod;

				 char plik[15];
				 pom = "file.txt";
				 strcpy(plik, pom.c_str());
				 do_kod = fopen(plik, "r");

              FILE* kod=fopen("zakodowany.txt","w");
              if(do_kod==NULL)
              {
                    cout<<"Plik \"zakodowany.txt\" nie istnieje, musial byc przeniesiony,\n";
                    cout<<" badz jego nazwa zostala zmieniona\n";
                    fclose(do_kod);
                    fclose(kod);
                    system("PAUSE");
                    return 0;
              }
              kodowanie(h,do_kod,kod);
              fclose(do_kod);
              fclose(kod);
              //system("PAUSE");
              //return 0;
         }
else if (wybor==2)
         {
              FILE*kod=fopen("zakodowany.txt","r");
              FILE*odkod=fopen("odkodowany.txt","w");
              sprawdzanie(h,kod,odkod);
              fclose(kod);
              fclose(odkod);
              //system("PAUSE");
              //return 0;
         }

else if (wybor==3)
         {
         cout<<"AUTORZY"<<endl<<endl;
         cout<<"Artur Olszczynski i Adrian Piesiak"<<endl;
         }

else if (wybor==4)
       {
         cout<<"Koniec!"<<endl;
         n=0;
         //system("PAUSE");
       }
else if (wybor!=1 || wybor!=2 || wybor!=3 || wybor!=4) cout<<"Z£A LICZBA"<<endl;


//}
}
system("PAUSE");
return 0;
}
