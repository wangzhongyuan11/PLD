var api = require("../../utils/api.js");
var FILE_EXTS = ["npz", "npy", "nii", "gz", "mhd", "raw"];

Page({
  data: {
    mode: "image",          // image | file
    image: null,            // {path, name}
    files: [],              // [{path, name, size}]
    spacingOn: false,
    spacingY: "",
    spacingX: "",
    submitting: false,
  },

  setMode: function (e) {
    this.setData({ mode: e.currentTarget.dataset.mode });
  },

  chooseImage: function () {
    var self = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["album"],
      success: function (res) {
        var f = res.tempFiles && res.tempFiles[0];
        if (!f) return;
        self.setData({ image: { path: f.tempFilePath, name: "camus.png", size: f.size } });
      },
      fail: function (err) { api.pickFailToast("选择图片", err); },
    });
  },

  clearImage: function () { this.setData({ image: null }); },

  chooseFiles: function () {
    var self = this;
    wx.chooseMessageFile({
      count: 4,
      type: "file",
      extension: FILE_EXTS,
      success: function (res) {
        var files = (res.tempFiles || []).map(function (f) {
          return { path: f.path, name: f.name, size: f.size };
        });
        self.setData({ files: files });
      },
      fail: function (err) { api.pickFailToast("选择文件", err); },
    });
  },

  removeFile: function (e) {
    var idx = Number(e.currentTarget.dataset.index);
    var files = this.data.files.slice();
    files.splice(idx, 1);
    this.setData({ files: files });
  },

  toggleSpacing: function (e) {
    this.setData({ spacingOn: e.detail.value.length > 0 });
  },

  onY: function (e) { this.setData({ spacingY: e.detail.value }); },
  onX: function (e) { this.setData({ spacingX: e.detail.value }); },

  submit: function () {
    var self = this;
    if (this.data.submitting) return;
    var files = [];
    var title = "";
    if (this.data.mode === "image") {
      if (!this.data.image) return;
      files.push({ path: this.data.image.path, name: this.data.image.name });
      title = this.data.image.name;
    } else {
      if (!this.data.files.length) return;
      files = this.data.files;
      title = files[0].name;
    }
    var params = { input_name: files[0].name };
    if (this.data.spacingOn) {
      var y = parseFloat(this.data.spacingY);
      var x = parseFloat(this.data.spacingX);
      if (!(y > 0) || !(x > 0)) {
        wx.showToast({ title: "像素间距需为正数（毫米）", icon: "none" });
        return;
      }
      params.spacing = [y, x];
    }
    this.setData({ submitting: true });
    api.submitTaskFlow({
      type: "camus",
      files: files,
      params: params,
      title: title,
    }).then(function (id) {
      wx.redirectTo({ url: "/pages/task/task?id=" + id });
    }).catch(function (err) {
      self.setData({ submitting: false });
      wx.showToast({ title: err.message || "提交失败", icon: "none" });
    });
  },
});
