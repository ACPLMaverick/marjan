package  
{
	import flash.display.Scene;
	import flash.display.MovieClip;
	import flash.display.Stage;
	import flash.utils.Dictionary;
	import flash.events.Event;
	import away3d.containers.View3D;
	import flash.geom.Vector3D;
	import away3d.entities.Mesh;
	import away3d.primitives.PlaneGeometry;
	import away3d.library.AssetLibrary;
	import flash.net.URLRequest;
	import away3d.events.LoaderEvent;
	import away3d.loaders.parsers.OBJParser;
	import away3d.library.utils.AssetLibraryIterator;
	import away3d.events.AssetEvent;
	import away3d.core.base.Geometry;
	import away3d.animators.IAnimationSet;
	import flash.xml.XMLDocument;
	import flash.net.URLLoader;
	import away3d.core.pick.PickingType;
	import flash.events.MouseEvent;
	import away3d.containers.Scene3D;
	import away3d.core.render.RendererBase;
	import away3d.materials.TextureMaterial;
	import away3d.textures.Texture2DBase;
	import away3d.loaders.parsers.Parsers;
	import away3d.loaders.parsers.ImageParser;
	import away3d.loaders.parsers.AWDParser;
	import fl.motion.Color;
	import flash.display.BitmapData;
	import flash.display.Bitmap;
	import flash.display.Loader;
	import flash.display.LoaderInfo;
	import away3d.textures.BitmapTexture;
	import flash.geom.Matrix;
	import away3d.containers.ObjectContainer3D;
	import away3d.textures.VideoTexture;
	import flash.display3D.textures.Texture;
	import away3d.textures.BitmapCubeTexture;
	import away3d.primitives.SkyBox;
	import away3d.lights.LightBase;
	import away3d.lights.PointLight;
	import away3d.materials.lightpickers.StaticLightPicker;
	import away3d.filters.BloomFilter3D;
	import away3d.lights.DirectionalLight;
	
	public class System 
	{
		private static var instance:System;
		
		public const CAMERA_SELECTOR_POSITON:Vector3D = new Vector3D(0.0, 200.0, -1000.0);
		public const CAMERA_MODELVIEW_POSITON:Vector3D = new Vector3D(-700.0, 200.0, -550.0);
		public const SHOWCASE_MODEL_POSITION:Vector3D = new Vector3D(-300.0, 200.0, -550.0);
		public const CAMERA_START_POSITON:Vector3D = new Vector3D(300.0, 70.0, -100.0);
		
		private var stage:Stage;
		
		private var graphics:View3D;
		private var filter:BloomFilter3D;
		private var camera:CustomCamera3D;
		private var lastCameraPos:Vector3D;
		private var lastCameraTgt:Vector3D;
		private var cameraMoved:Boolean = false;
		
		private var mouseController:MouseController;
		
		private var loadScreen:LoadScreenContainer;
		
		private var objects:Dictionary;
		private var models:Dictionary;
		private var textures:Dictionary;
		private var materials:Dictionary;
		private var pictures:Dictionary;
		private var lights:Array;
		
		private var slp:StaticLightPicker;
		private var sld:StaticLightPicker;
		
		private var selector:Selector;
		
		private var xmlLoader:URLLoader;
		private var picLoaders:Array;
		private var xml:XML;
		
		private var modelsToLoad:uint = 0;
		private var modelsLoaded:Boolean = false;
		
		private var overviewSetup:Boolean = false;
		
		private var objectOnHold:SceneObject = null;
		
		public static const MODE_SELECTION:uint = 0;
		public static const MODE_OVERVIEW:uint = 1;
		public static const MODE_MODELVIEW:uint = 2;
		
		public var CurrentMode:uint = System.MODE_SELECTION;
		
		public function Initialize(stage:Stage) : void
		{
			this.stage = stage;
			
			
			this.loadScreen = new LoadScreenContainer(stage);
			this.stage.addChild(this.loadScreen);
			
			this.camera = new CustomCamera3D();
			this.graphics = new View3D(new Scene3D(), camera);
			this.stage.addChild(this.graphics);
			this.graphics.render();
			
			this.graphics.mousePicker = PickingType.RAYCAST_BEST_HIT;
			this.graphics.antiAlias = 4;
			this.graphics.backgroundColor = 0;
			this.filter = new BloomFilter3D(4, 4, 0.6, 2, 3);
			this.graphics.filters3d = [this.filter];
			
			this.stage.addEventListener(Event.ENTER_FRAME, Update);
			
			this.objects = new Dictionary();
			this.models = new Dictionary();
			this.textures = new Dictionary();
			this.materials = new Dictionary();
			this.pictures = new Dictionary();
			this.lights = new Array();
			
			this.picLoaders = new Array();
			
			/*this.mouseController = new MouseController();
			this.mouseController.Initialize(this, this.stage);*/
			var mc:MouseController = MouseController.getInstance();
			mc.Initialize(this.stage);
			
			LoadContent();
	
			///////
			// setup here
			//
			graphics.camera.position = CAMERA_START_POSITON;
			graphics.camera.lookAt(new Vector3D);
			
			// setting up lights
			var pointLight:PointLight = new PointLight();
			this.lights.push(pointLight);
			pointLight.color = 0xFFFFFFFF;
			pointLight.diffuse = 0.8;
			pointLight.fallOff = 600;
			pointLight.radius = 200;
			pointLight.y = 300;
			
			var dirLight:DirectionalLight = new DirectionalLight(-1, -1, 1);
			this.lights.push(dirLight);
			dirLight.color = 0xFFFFFCFC;
			dirLight.ambientColor = 0xFFFFF0F0;
			dirLight.ambient = 0.4;
			dirLight.diffuse = 1;
			
			this.slp = new StaticLightPicker([pointLight]);
			this.sld = new StaticLightPicker([dirLight]);
			
			///////
		}
		
		public function Update(event:Event) : void
		{	
			if(CurrentMode == System.MODE_SELECTION)
			{
				if(modelsToLoad > 0)
				{
					if(!loadScreen.Visible)
						loadScreen.Visible = true;
				}
				else if(!modelsLoaded)	// ONLY ONCE
				{
					this.modelsLoaded = true;
					
					// generate objects from geometries
					selector = new Selector("Selector", xml);
					selector.position = new Vector3D(0.0, 400, 0.0);
					objects["Selector"] = selector;
					
					// generating skybox
					var cubeTexture:BitmapCubeTexture = new BitmapCubeTexture((textures["textures/SkyBox/posx.jpg"] as BitmapTexture).bitmapData,
																			  (textures["textures/SkyBox/negx.jpg"] as BitmapTexture).bitmapData,
																			  (textures["textures/SkyBox/posy.jpg"] as BitmapTexture).bitmapData,
																			  (textures["textures/SkyBox/negy.jpg"] as BitmapTexture).bitmapData,
																			  (textures["textures/SkyBox/posz.jpg"] as BitmapTexture).bitmapData,
																			  (textures["textures/SkyBox/negz.jpg"] as BitmapTexture).bitmapData);
					var skyBox:SkyBox = new SkyBox(cubeTexture);
					
					// adding shit to scene
					(objects["Pyramid"] as SceneObject).material.lightPicker = this.slp;
					(objects["Cylinder001"] as SceneObject).material.lightPicker = this.slp;
					(objects["Terrain"] as SceneObject).material.lightPicker = this.slp;
					graphics.scene.addChild(objects["Pyramid"]);
					graphics.scene.addChild(objects["Cylinder001"]);
					graphics.scene.addChild(objects["Terrain"]);
					graphics.scene.addChild(skyBox);
					graphics.scene.addChild(selector);
					
					for each(var light:LightBase in this.lights)
					{
						graphics.scene.addChild(light);
						trace("added");
					}
					
					// setting up camera
					camera.SmoothMovement(CAMERA_SELECTOR_POSITON, selector.position, 2500);
					
					// setting up other things
					(objects["Pyramid"] as SceneObject).rotationY = 57;
					//selector.StartAnimation();
				}
				else
				{
					// DO EVERYTHING FOR LOADED SCENE
					if(loadScreen.Visible)
						loadScreen.Visible = false;
					
					for each(var value3:SceneObject in objects)
					{
						value3.Update();
					}
				
					//(objects["Selector"] as SceneObject).rotationY += 3;
					graphics.render();
				}
			}
			else if(CurrentMode == System.MODE_OVERVIEW)
			{
				if(modelsToLoad > 0)
				{
					if(!loadScreen.Visible)
						loadScreen.Visible = true;
				}
				else
				{
					if(loadScreen.Visible)
						loadScreen.Visible = false;
						
					for each(var value4:SceneObject in objects)
					{
						value4.Update();
					}
				}
				graphics.render();
			}
			else if(CurrentMode == System.MODE_MODELVIEW)
			{
				if(modelsToLoad > 0)
				{
					if(!loadScreen.Visible)
						loadScreen.Visible = true;
				}
				else
				{
					if(loadScreen.Visible)
						loadScreen.Visible = false;
						
					for each(var value5:SceneObject in objects)
					{
						value5.Update();
					}
					
					graphics.render();
				}
			}
			
			MouseController.getInstance().Update();
			this.camera.Update();
		}
		
		private function LoadContent() : void
		{
			/// loading objs
			AssetLibrary.enableParser(AWDParser);
			AssetLibrary.enableParser(ImageParser);
			AssetLibrary.addEventListener(AssetEvent.ASSET_COMPLETE, LoadHandler);
			LoadToLibrary("./models/Pyramid.awd", 2);
			LoadToLibrary("./models/Terrain.awd", 1);
			LoadToLibrary("./models/InfoPlane.awd", 8);
			LoadTexture("textures/PlayVideoTexture.png");
			LoadTexture("textures/SkyBox/negx.jpg");
			LoadTexture("textures/SkyBox/negy.jpg");
			LoadTexture("textures/SkyBox/negz.jpg");
			LoadTexture("textures/SkyBox/posx.jpg");
			LoadTexture("textures/SkyBox/posy.jpg");
			LoadTexture("textures/SkyBox/posz.jpg");
			
			/// loading other objects
			
			//objects["Terrain"] = new SceneObject(new PlaneGeometry(5000, 5000, 1, 1, true, false), null, "Terrain");
			/// loading materials
			
			/// loading info
			LoadInfoFromXML("./info.xml");
			
			///////////////////////////////////////
		}
		
		public function LoadToLibrary(path:String, modCount:uint)
		{
			AssetLibrary.load(new URLRequest(path), null, ParseObjectPath(path));
			this.modelsToLoad += modCount;
		}
		
		public function LoadTexture(path:String)
		{
			var picLoader:Loader = new Loader();
			picLoaders.push(picLoader);
			picLoader.contentLoaderInfo.addEventListener(Event.COMPLETE, TextureLoadHandler);
			picLoader.load(new URLRequest(path));
			++this.modelsToLoad;
		}
		
		public function LoadPicture(path:String)
		{
			var picLoader:Loader = new Loader();
			picLoaders.push(picLoader);
			picLoader.contentLoaderInfo.addEventListener(Event.COMPLETE, PicLoadHandler);
			picLoader.load(new URLRequest(path));
			++this.modelsToLoad;
		}
		
		public function GetMaterialFromVideoPath(path:String, thumbPath:String) : TextureMaterial
		{
			var bm:BitmapData = ((pictures[thumbPath] as TextureMaterial).texture as BitmapTexture).bitmapData;
			var vt:CustomVideoTexture = new CustomVideoTexture(path, 512, 512, true, false, bm, null);
			var newTM:TextureMaterial = new TextureMaterial(vt);
			materials[path] = newTM;
			return newTM;
		}
		
		private function LoadHandler(event:AssetEvent) : void
		{
			if(event.asset is Mesh)
			{
				var mesh:Mesh = event.asset as Mesh;
				mesh.geometry.scale(10);
				models[mesh.name + "Gm"] = mesh.geometry.clone();
				materials[mesh.material.name] = mesh.material;
				
				var texCount:uint = 0;
				if(mesh.material is TextureMaterial)
				{
					var texMat:TextureMaterial = mesh.material as TextureMaterial;
					if(texMat.texture != null)
					{
						textures[texMat.texture.name] = texMat.texture;
						++texCount;
					}
						
					if(texMat.specularMap != null)
					{
						textures[texMat.specularMap.name] = texMat.specularMap;
						++texCount;
					}
						
					if(texMat.normalMap != null)
					{
						textures[texMat.normalMap.name] = texMat.normalMap;
						++texCount;
					}
					
					//mesh.material.lightPicker = this.slp;
				}
				
				objects[mesh.name] = new SceneObject(mesh.geometry, mesh.material, mesh.name);
				
				// not yet
				//graphics.scene.addChild(objects[mesh.name]);
				
				trace("System: AWD " + mesh.name + " loaded. Texture count: " + texCount.toString());
				--this.modelsToLoad;
			}
			else if(event.asset is ObjectContainer3D)
			{
				var cont:ObjectContainer3D = event.asset as ObjectContainer3D;
				trace(cont.numChildren);
			}
		}
		
		private function PicLoadHandler(event:Event) : void
		{
			--this.modelsToLoad;
			
			var projectPath:String = (event.currentTarget as LoaderInfo).loaderURL;
			var fullPath:String = (event.currentTarget as LoaderInfo).url;
			
			var pPSplit:Array = projectPath.split('/');
			var pPSplitL:uint = pPSplit.length - 1;
			var projectPathWO:String = "";
			for(var i:uint = 0; i < pPSplitL; ++i)
			{
				projectPathWO += pPSplit[i] + "/";
			}
			fullPath = fullPath.substring(projectPathWO.length, fullPath.length);
			
			var bm:BitmapData = ((event.currentTarget as LoaderInfo).content as Bitmap).bitmapData;
			var nbm:BitmapData = new BitmapData(2048, 2048, true, 0);
			var mt:Matrix = new Matrix();
			mt.scale(1.28, 2.275555);
			nbm.draw(bm, mt);
			var tm:TextureMaterial = new TextureMaterial(new BitmapTexture(nbm, true));
			tm.alphaBlending = true;
			tm.alphaThreshold = 0.01;
															
			pictures[fullPath] = tm;
			
			trace("System: picture loaded: " + fullPath);
		}
		
		private function TextureLoadHandler(event:Event) : void
		{
			--this.modelsToLoad;
			
			var projectPath:String = (event.currentTarget as LoaderInfo).loaderURL;
			var fullPath:String = (event.currentTarget as LoaderInfo).url;
			
			var pPSplit:Array = projectPath.split('/');
			var pPSplitL:uint = pPSplit.length - 1;
			var projectPathWO:String = "";
			for(var i:uint = 0; i < pPSplitL; ++i)
			{
				projectPathWO += pPSplit[i] + "/";
			}
			fullPath = fullPath.substring(projectPathWO.length, fullPath.length);
			
			var bm:BitmapData = ((event.currentTarget as LoaderInfo).content as Bitmap).bitmapData;
			var tex:BitmapTexture = new BitmapTexture(bm, true);
			textures[fullPath] = tex;
			
			trace("System: texture loaded: " + fullPath);
		}
		
		private function ParseObjectPath(path:String) : String
		{
			var array:Array = path.split('/');
			var myName:String = (array[array.length - 1] as String);
			var yetAnotherArray:Array = myName.split('.');
			return (yetAnotherArray[0] as String);
		}
		
		private function LoadInfoFromXML(path:String)
		{
			this.xmlLoader = new URLLoader(new URLRequest(path));
			this.xmlLoader.addEventListener(Event.COMPLETE, InfoLoadHandler);
			++this.modelsToLoad;
		}
		
		private function InfoLoadHandler(e:Event)
		{
			trace("System: XML with info data loaded.");
			this.xml = new XML(this.xmlLoader.data);
			
			--this.modelsToLoad;
			
			// load all pictures in xml
			var pList:XMLList = xml.asset;
			var pListLength:uint = pList.length();
			for(var i:uint = 0; i < pListLength; ++i)
			{
				if("pics" in pList[i])
				{
					for each(var pic:XML in pList[i].pics.pic)
					{
						if(pic.hasOwnProperty("@path"))
						{
							LoadPicture(pic.@path);
						}
					}
				}
				
				if("videos" in pList[i])
				{
					for each(var vid:XML in pList[i].videos.video)
					{
						if(vid.hasOwnProperty("@thumb"))
						{
							LoadPicture(vid.@thumb);
						}
					}
				}
			}
		}
		
		public function SetMode(mode:uint)
		{
			this.CurrentMode = mode;
			
			switch(mode)
			{
				case System.MODE_SELECTION:
					camera.SmoothMovement(CAMERA_SELECTOR_POSITON, selector.position, 750);
					selector.EnableInteractivity();
					selector.CurrentlySelected.DisableFields();
					camera.Locked = true;
				break;
				
				case System.MODE_OVERVIEW:
					var vecPos:Vector3D = selector.CurrentlySelected.position;
					vecPos.y += selector.position.y;
					vecPos.z -= 300.0;
					var vecTgt:Vector3D = selector.CurrentlySelected.position;
					vecTgt.y += selector.position.y;
					camera.SmoothMovement(vecPos, vecTgt, 500);
					
					selector.DisableInteractivity();
					selector.CurrentlySelected.EnableFields();
					if(selector.CurrentlySelected.ModelShown)
					{
						selector.CurrentlySelected.HideModel();
					}
					camera.Locked = true;
				break;
				
				case System.MODE_MODELVIEW:
					selector.DisableInteractivity();
					selector.CurrentlySelected.DisableFields();
					camera.SmoothMovement(CAMERA_MODELVIEW_POSITON, SHOWCASE_MODEL_POSITION, 700);
					camera.Locked = false;
				break;
			}
		}
		
		public function CameraSingleMove(pos:Vector3D, tgt:Vector3D, timeMS:uint)
		{
			if(!this.cameraMoved)
			{
				this.lastCameraPos = camera.position.clone();
				this.lastCameraTgt = camera.Target.clone();
				this.cameraMoved = true;
				this.camera.SmoothMovement(pos, tgt, timeMS);
			}
			else
			{
				throw new Error("Camera is already moved");
			}
		}
		
		public function CameraSingleUnmove(timeMS:uint)
		{
			this.selector.GetThatShitFixed();
			if(this.cameraMoved)
			{
				this.cameraMoved = false;
				this.camera.SmoothMovement(lastCameraPos, lastCameraTgt, timeMS);
			}
			else
			{
				throw new Error("Camera is not moved");
			}
		}
		
		public function AddToScene(s:SceneObject) : void
		{
			this.graphics.scene.addChild(s);
		}
		
		public function DispatchClick(e:MouseEvent) : void
		{
			
		}
		
		public function DispatchHold(e:MouseEvent) : void
		{
			
		}
		
		public static function Vector3DLerp(a:Vector3D, b:Vector3D, lerp:Number) : Vector3D
		{
			var lerped:Vector3D = new Vector3D();
			lerped.x = a.x + (b.x - a.x) * lerp;
			lerped.y = a.y + (b.y - a.y) * lerp;
			lerped.z = a.z + (b.z - a.z) * lerp;
			lerped.w = a.w + (b.w - a.w) * lerp;
			return lerped;
		}
		
		public static function getInstance() : System
		{
			if(System.instance == null)
			{
				System.instance = new System();
			}
			return System.instance;
		}
		
		public function get Objects() : Dictionary
		{
			return objects;
		}
		
		public function get Models() : Dictionary
		{
			return models;
		}
		
		public function get Textures() : Dictionary
		{
			return textures;
		}
		
		public function get Materials() : Dictionary
		{
			return materials;
		}
		
		public function get Pictures() : Dictionary
		{
			return pictures;
		}
		
		public function get Graphics() : View3D
		{
			return graphics;
		}
		
		public function get MyStage() : Stage
		{
			return stage;
		}
		
		public function get Cam() : CustomCamera3D
		{
			return this.camera;
		}
		
		public function get IsCameraMoved() : Boolean
		{
			return this.cameraMoved;
		}
		
		public function get Slp() : StaticLightPicker
		{
			return this.slp;
		}
		
		public function get Sld() : StaticLightPicker
		{
			return this.sld;
		}
	}
}
