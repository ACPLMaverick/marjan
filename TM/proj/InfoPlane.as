package  
{
	import away3d.entities.Mesh;
	import away3d.core.base.Geometry;
	import away3d.materials.MaterialBase;
	import away3d.events.MouseEvent3D;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import flash.geom.Vector3D;
	import away3d.core.math.Vector3DUtils;
	import away3d.core.base.SubGeometry;
	import away3d.bounds.NullBounds;
	import away3d.events.LoaderEvent;
	import away3d.library.AssetLibrary;
	
	public class InfoPlane extends SceneObject
	{
		private var geometries:Array;
		private var baseMaterials:Array;
		
		private var objName:String;
		private var objDesc:String;
		private var images:Array;
		private var videos:Array;
		private var associatedModel:ShowcaseContainer;
		private var associatedModelPath:String;
		private var associatedModelPartPaths:Array;
		private var associatedModelScale:Number = 1;
		private var modelShown:Boolean = false;
		
		private var fieldTitle:SceneObject;
		private var fieldDesc:SceneObject;
		public var fieldViewer:SceneObject;
		private var fieldBtnShowDesc:SceneObject;
		private var fieldBtnShowModel:SceneObject;
		private var fieldBtnLeft:SceneObject;
		private var fieldBtnRight:SceneObject;
		
		private var fieldsEnabled:Boolean = false;
		
		public function InfoPlane(geometries:Array, materials:Array,
								  objName:String, objDesc:String, images:Array, videos:Array = null, associatedModelPath:String = null,
								  associatedModelPartPaths:Array = null, associatedMeshScale:Number = 1, name:String = null) 
		{	
			super(geometries[0] as Geometry, materials[0] as MaterialBase, name, true);
			
			this.baseMaterials = materials;
			this.geometries = geometries;
			this.objName = objName;
			this.objDesc = objDesc;
			this.images = images;
			this.videos = videos;
			this.associatedModelPath = associatedModelPath;
			this.associatedModelPartPaths = associatedModelPartPaths;
			this.associatedModelScale = associatedMeshScale;
			GenerateFields();
		}

		public override function Update() : void
		{
			super.Update();
			// do nothing
			if(associatedModel != null && modelShown)
			{
				associatedModel.Update();
			}
		}
		
		public override function ActionClick(me:MouseEvent3D) : void
		{
			if((this.parent as Selector).CurrentlySelected == this)
			{
				System.getInstance().SetMode(System.MODE_OVERVIEW);
				return;
			}
			(this.parent as Selector).Select(this);
		}
		
		public override function ActionHoldIn() : void
		{
			if(holdMe != null)
			{
				var dot:Number = GetPosCamDot();
			
				var dir:int = 1;
				if(dot < 0)
				{
					dir = -1;
				}
				var selector:Selector = (this.parent as Selector);
				//selector.StartRotatingAsMouse(dir);
				selector.SelectAsMouse(dir);
			}
		}
		
		public override function ActionHoldOut(me:MouseEvent3D) : void
		{
			if(this.holdMe != null)
			{
				//(this.parent as Selector).StopRotatingAsMouse();
				//(this.parent as Selector).SelectClosest();
				holdMe = null;
			}
		}
		
		public override function ActionHoverIn(me:MouseEvent3D) : void
		{
			
		}
		
		public override function ActionHoverOut(me:MouseEvent3D) : void
		{
			if(this.holdMe != null)
			{
				//(this.parent as Selector).StopRotatingAsMouse();
				//(this.parent as Selector).SelectClosest();
				holdMe = null;
			}
		}
		
		private function GenerateFields() : void
		{
			this.fieldTitle = new FieldTitle(geometries[1], null, this.objName);
			
			this.fieldDesc = new FieldDescription(geometries[2] as Geometry, baseMaterials[2] as MaterialBase, null, this.objDesc);
			
			this.fieldViewer = new FieldViewer(geometries[3] as Geometry, images, videos, null);
			
			this.fieldBtnShowDesc = new FieldToggleDescButton(geometries[4] as Geometry, this.fieldDesc);
			
			if(this.associatedModelPath != null)
			{
				this.fieldBtnShowModel = new FieldToggleModelButton(geometries[5] as Geometry, this);
			}
			
			this.fieldBtnLeft = new FieldChangeButton(geometries[6] as Geometry, baseMaterials[6] as MaterialBase,
													  this.fieldViewer as FieldViewer, FieldChangeButton.SIDE_LEFT);
			this.fieldBtnRight = new FieldChangeButton(geometries[7] as Geometry, baseMaterials[7] as MaterialBase,
													  this.fieldViewer as FieldViewer, FieldChangeButton.SIDE_RIGHT);
			
			this.addChild(this.fieldTitle);
			this.addChild(this.fieldDesc);
			this.addChild(this.fieldViewer);
			this.addChild(this.fieldBtnShowDesc);
			if(this.associatedModelPath != null)
			{
				this.addChild(this.fieldBtnShowModel);
			}
			this.addChild(this.fieldBtnLeft);
			this.addChild(this.fieldBtnRight);
			
			fieldsEnabled = true;
			DisableFields();
		}
		
		public function ShowModel() : void
		{
			if(associatedModel == null)
			{
				RequestModel();
				return;
			}
			
			modelShown = true;
			associatedModel.visible = true;
			System.getInstance().SetMode(System.MODE_MODELVIEW);
		}
		
		public function HideModel() : void
		{
			if(associatedModel == null)
			{
				// impossibiru
				return;
			}
			
			modelShown = false;
			if(this.fieldBtnShowModel != null)
				this.fieldBtnShowModel.ActionClick(null);
			associatedModel.visible = false;
		}
		
		private function RequestModel() : void
		{
			this.DisableFields();
			AssetLibrary.addEventListener(LoaderEvent.RESOURCE_COMPLETE, RequestCompletedHandler);
			System.getInstance().LoadToLibrary(this.associatedModelPath, this.associatedModelPartPaths.length);
		}
		
		private function RequestCompletedHandler(e:LoaderEvent) : void
		{
			this.EnableFields();
			AssetLibrary.removeEventListener(LoaderEvent.RESOURCE_COMPLETE, RequestCompletedHandler);
			
			associatedModel = new ShowcaseContainer(this.name + "Model");
			var modArray:Array = new Array();
			for each(var str:String in associatedModelPartPaths)
			{
				modArray.push(System.getInstance().Objects[str]);
			}
			
			for each(var mod:SceneObject in modArray)
			{
				mod.EnableInteractivity();
				mod.material.lightPicker = System.getInstance().Sld;
				associatedModel.addChild(mod);
			}
			
			associatedModel.visible = false;
			associatedModel.position = System.getInstance().SHOWCASE_MODEL_POSITION;
			associatedModel.scale(this.associatedModelScale);
			System.getInstance().AddToScene(associatedModel);
			
			ShowModel();
		}
		
		public function EnableFields() : void
		{
			if(!fieldsEnabled)
			{
				fieldsEnabled = true;
				
				this.fieldTitle.mouseEnabled = true;
				this.fieldDesc.mouseEnabled = true;
				this.fieldViewer.mouseEnabled = true;
				this.fieldBtnShowDesc.mouseEnabled = true;
				if(fieldBtnShowModel != null)
					this.fieldBtnShowModel.mouseEnabled = true;
				this.fieldBtnLeft.mouseEnabled = true;
				this.fieldBtnRight.mouseEnabled = true;
			}
		}
		
		public function DisableFields() : void
		{
			if(fieldsEnabled)
			{
				fieldsEnabled = false;
				
				this.fieldTitle.mouseEnabled = false;
				this.fieldDesc.mouseEnabled = false;
				this.fieldViewer.mouseEnabled = false;
				(this.fieldViewer as FieldViewer).StopVideos();
				this.fieldBtnShowDesc.mouseEnabled = false;
				if(fieldBtnShowModel != null)
					this.fieldBtnShowModel.mouseEnabled = false;
				this.fieldBtnLeft.mouseEnabled = false;
				this.fieldBtnRight.mouseEnabled = false;
			}
		}
		
		public function get ObjName() : String
		{
			return objName;
		}
		
		public function get ObjDesc() : String
		{
			return objDesc;
		}
		
		public function get Images() : Array
		{
			return images;
		}
		
		public function get Videos() : Array
		{
			return videos;
		}
		
		public function get AssociatedModel() : Mesh
		{
			return associatedModel;
		}
		
		public function get ModelShown() : Boolean
		{
			return modelShown;
		}
	
		private function GetPosCamDot() : Number
		{
			var myPos:Vector3D = position.clone();
			myPos.normalize();
			var camPos:Vector3D = (System.getInstance().Graphics.camera.position).clone();
			camPos.normalize();
			var dot:Number = (myPos).dotProduct(camPos);
			return dot;
		}
	}
	
}
