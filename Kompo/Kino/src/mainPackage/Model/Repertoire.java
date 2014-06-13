package mainPackage.Model;

import java.util.ArrayList;
import java.util.Date;

/**
 * 
 * Reprezentuje ca³y repertuar, przechowuje kolekcjê seansów
 * Automatycznie synchronizuje siê z baz¹ danych.
 *
 */
public class Repertoire {
	
	private ArrayList<Seance> seanceList;
	private ArrayList<Film> filmList;
	private DBController myDBController;
	
	public Repertoire()
	{
		this.seanceList = new ArrayList<Seance>();
		this.myDBController = new DBController();
		this.seanceList = this.myDBController.getWholeRepertoire();
		this.filmList = this.myDBController.getAllFilms();
	}
	
	public void add(Seance seance)
	{
		seanceList.add(seance);
		myDBController.addSeance(seance);
	}
	
	public void delete(int i)
	{
		Seance toDelete = seanceList.get(i);
		seanceList.remove(i);
		myDBController.deleteSeance(toDelete.getDateAsString());
	}
	
	public void update(Seance seance, int i)
	{
		myDBController.updateSeance(seance, seanceList.get(i).getDateAsString());
		this.seanceList = this.myDBController.getWholeRepertoire();
	}
	
	public Seance get(int i)
	{
		return seanceList.get(i);
	}
	
	public ArrayList<Seance> get()
	{
		return seanceList;
	}
	
	//////////////////////////////////////
	
	public ArrayList<Film> getFilms()
	{
		return filmList;
	}

	public void addFilm(Film film)
	{
		filmList.add(film);
		myDBController.addFilm(film);
	}
	
	public void deleteFilm(int i)
	{
		Film toDelete = filmList.get(i);
		filmList.remove(i);
		myDBController.deleteFilm(toDelete.getTitle());
	}
	
	public void updateFilm(Film film, int i)
	{
		myDBController.updateFilm(film, filmList.get(i).getTitle());
		this.filmList = this.myDBController.getAllFilms();
	}
	
	public Film getFilm(int i)
	{
		return filmList.get(i);
	}
}
